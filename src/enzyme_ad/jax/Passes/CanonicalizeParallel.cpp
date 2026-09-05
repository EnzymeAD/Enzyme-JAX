//===- CanonicalizeParallel.cpp - Per-op parallel canonicalization --------===//
//
// The raising pipeline canonicalizes the module many times, and the stock
// canonicalizer walks it as one serial unit: one greedy driver seeds a
// worklist with every operation in the module, and each driver iteration
// re-simplifies every region in it. On a translation unit whose module holds
// thousands of functions that is the pipeline's dominant cost.
//
// This pass canonicalizes each top-level region-holding op as its own unit,
// in parallel. Confinement is as important as the parallelism: region
// simplification -- unreachable-block elimination and the region liveness
// fixpoint -- runs per driver iteration over the scope the driver was given,
// so giving each op its own driver keeps one op's churn from re-walking
// every other op's regions. Regionless children -- globals and declarations
// -- are canonicalized together as one cheap batch.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/Support/KnownBits.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZEPARALLELPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using llvm::KnownBits;

namespace {

// A truncation that discards every bit an or set sees through the or. This is
// how a kernel launch reads its dimensions back out of clang's packed dim3:
// grid.y lands in the high half of an i64 and the launch takes the low half,
// and the bits in between are what tie the grid to the size it was computed
// from.
struct TruncOrConst : public OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp trunc,
                                PatternRewriter &rewriter) const override {
    auto intTy = dyn_cast<IntegerType>(trunc.getType());
    if (!intTy)
      return failure();
    auto ori = trunc.getIn().getDefiningOp<arith::OrIOp>();
    if (!ori)
      return failure();
    APInt cst;
    Value other;
    if (matchPattern(ori.getRhs(), m_ConstantInt(&cst)))
      other = ori.getLhs();
    else if (matchPattern(ori.getLhs(), m_ConstantInt(&cst)))
      other = ori.getRhs();
    else
      return failure();
    if (!cst.extractBits(intTy.getWidth(), 0).isZero())
      return failure();
    rewriter.modifyOpInPlace(trunc,
                             [&] { trunc.getInMutable().assign(other); });
    return success();
  }
};

// The symmetric/nonsymmetric slice picks of MFEM kernels reach MLIR as a
// select over geps off one base: clang folds the constant slice indices
// into the addressing, and the conditional survives as pointer control
// flow. Sink the select back into the index so the address chain stays a
// single gep, which the view rebasing canonicalizations can see through.
struct SelectOfSameBaseGEPs : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const override {
    if (!isa<LLVM::LLVMPointerType>(sel.getType()))
      return failure();
    auto gepT = sel.getTrueValue().getDefiningOp<LLVM::GEPOp>();
    auto gepF = sel.getFalseValue().getDefiningOp<LLVM::GEPOp>();
    // A side that is the base itself stepped zero elements of the other's
    // type.
    int bare = -1;
    if (!gepT && gepF && sel.getTrueValue() == gepF.getBase()) {
      gepT = gepF;
      bare = 0;
    } else if (!gepF && gepT && sel.getFalseValue() == gepT.getBase()) {
      gepF = gepT;
      bare = 1;
    }
    if (!gepT || !gepF)
      return failure();
    if (gepT.getBase() != gepF.getBase() ||
        gepT.getElemType() != gepF.getElemType())
      return failure();
    if (gepT.getIndices().size() != 1 || gepF.getIndices().size() != 1)
      return failure();

    auto dynT = dyn_cast_if_present<Value>(gepT.getIndices()[0]);
    auto dynF = dyn_cast_if_present<Value>(gepF.getIndices()[0]);
    Type idxTy;
    if (dynT && dynF) {
      if (dynT.getType() != dynF.getType())
        return failure();
      idxTy = dynT.getType();
    } else if (dynT) {
      idxTy = dynT.getType();
    } else if (dynF) {
      idxTy = dynF.getType();
    } else {
      idxTy = rewriter.getI64Type();
    }

    auto materialize = [&](LLVM::GEPOp gep, Value dyn, bool zero) -> Value {
      if (dyn && !zero)
        return dyn;
      int64_t v = zero ? 0
                       : cast<IntegerAttr>(gep.getIndices()[0])
                             .getValue()
                             .getSExtValue();
      Value c = arith::ConstantOp::create(rewriter, gep.getLoc(),
                                          IntegerAttr::get(idxTy, v));
      return c;
    };
    Value idxT = materialize(gepT, dynT, bare == 0);
    Value idxF = materialize(gepF, dynF, bare == 1);
    Value idx = arith::SelectOp::create(rewriter, sel.getLoc(),
                                        sel.getCondition(), idxT, idxF);
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        sel, sel.getType(), gepT.getElemType(), gepT.getBase(),
        SmallVector<LLVM::GEPArg>{idx},
        gepT.getNoWrapFlags() & gepF.getNoWrapFlags());
    return success();
  }
};

// Clang lowers every CUDA __shared__ access through one generic-izing
// addrspacecast, and all gep arithmetic happens on the generic pointer.
// Sink the cast toward its uses - the cast does not change the pointed-at
// bytes, so the offsets are identical in either space - until it dies at a
// pointer2memref, whose view type already tolerates the space mismatch.
struct SinkAddrSpaceCastThroughGEP : public OpRewritePattern<LLVM::GEPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::GEPOp gep,
                                PatternRewriter &rewriter) const override {
    auto asc = gep.getBase().getDefiningOp<LLVM::AddrSpaceCastOp>();
    if (!asc)
      return failure();
    auto srcTy = cast<LLVM::LLVMPointerType>(asc.getArg().getType());
    SmallVector<LLVM::GEPArg> args;
    for (auto idx : gep.getIndices()) {
      if (auto v = dyn_cast_if_present<Value>(idx))
        args.push_back(v);
      else
        args.push_back(cast<IntegerAttr>(idx).getValue().getSExtValue());
    }
    auto newGep = LLVM::GEPOp::create(
        rewriter, gep.getLoc(),
        LLVM::LLVMPointerType::get(gep.getContext(), srcTy.getAddressSpace()),
        gep.getElemType(), asc.getArg(), args, gep.getNoWrapFlags());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(gep, gep.getType(),
                                                       newGep);
    return success();
  }
};

struct Pointer2MemrefOfAddrSpaceCast
    : public OpRewritePattern<enzymexla::Pointer2MemrefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::Pointer2MemrefOp p2m,
                                PatternRewriter &rewriter) const override {
    auto asc = p2m.getSource().getDefiningOp<LLVM::AddrSpaceCastOp>();
    if (!asc)
      return failure();
    rewriter.replaceOpWithNewOp<enzymexla::Pointer2MemrefOp>(p2m, p2m.getType(),
                                                             asc.getArg());
    return success();
  }
};

/// Builds an if of the same flavour and condition with new result types.
static scf::IfOp createLikeIf(PatternRewriter &rewriter, scf::IfOp ifOp,
                              TypeRange types) {
  return scf::IfOp::create(rewriter, ifOp.getLoc(), types, ifOp.getCondition(),
                           /*withElseRegion=*/true);
}
static affine::AffineIfOp createLikeIf(PatternRewriter &rewriter,
                                       affine::AffineIfOp ifOp,
                                       TypeRange types) {
  return affine::AffineIfOp::create(rewriter, ifOp.getLoc(), types,
                                    ifOp.getIntegerSet(), ifOp.getOperands(),
                                    /*withElseRegion=*/true);
}

static void createLikeYield(PatternRewriter &rewriter, Location loc, scf::IfOp,
                            ValueRange values) {
  scf::YieldOp::create(rewriter, loc, values);
}
static void createLikeYield(PatternRewriter &rewriter, Location loc,
                            affine::AffineIfOp, ValueRange values) {
  affine::AffineYieldOp::create(rewriter, loc, values);
}

/// An op over what a branch chose between constants is the branch choosing
/// between the op's own results: the op rides into the arms, where it meets a
/// constant and folds. What a branch chose then reaches its user as the
/// branch's own result rather than at the far end of some arithmetic, which
/// is how an index reaches an access for split-branched-accesses.
template <typename OpT, typename IfT>
struct SinkThroughIfOfConstants : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    // One operand comes from the branch and the rest are constants, so each
    // arm folds to a constant of its own.
    OpResult branched;
    for (Value operand : op->getOperands()) {
      Attribute cst;
      if (matchPattern(operand, m_Constant(&cst)))
        continue;
      auto res = dyn_cast<OpResult>(operand);
      if (branched || !res || !isa<IfT>(res.getOwner()))
        return failure();
      branched = res;
    }
    // A second user would keep the branch alive and pay for it twice.
    if (!branched || !branched.hasOneUse())
      return failure();

    auto ifOp = cast<IfT>(branched.getOwner());
    if (ifOp->getRegion(1).empty())
      return failure();
    // The regions of both flavours of if are the arms, in order.
    unsigned resultNo = branched.getResultNumber();
    Value arms[2];
    for (unsigned arm = 0; arm < 2; ++arm) {
      arms[arm] =
          ifOp->getRegion(arm).front().getTerminator()->getOperand(resultNo);
      Attribute cst;
      if (!matchPattern(arms[arm], m_Constant(&cst)))
        return failure();
    }

    // Built where the op stands, so what the branch is taken over -- in hand
    // before the branch -- is in hand here too.
    auto newIf = createLikeIf(rewriter, ifOp, op->getResultTypes());
    for (unsigned arm = 0; arm < 2; ++arm) {
      rewriter.setInsertionPointToStart(&newIf->getRegion(arm).front());
      Operation *chosen = rewriter.clone(*arms[arm].getDefiningOp());
      IRMapping map;
      map.map(branched, chosen->getResult(0));
      Operation *cloned = rewriter.clone(*op.getOperation(), map);
      createLikeYield(rewriter, op.getLoc(), newIf, cloned->getResults());
    }
    rewriter.replaceOp(op, newIf->getResults());
    return success();
  }
};

/// The select counterpart: an op over a select between constants is the
/// select between the op over each arm, where it meets a constant and folds.
/// A select costs nothing to keep, so a second user of it is no reason to
/// leave the op where it is.
template <typename OpT>
struct SinkThroughSelectOfConstants : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    arith::SelectOp sel;
    for (Value operand : op->getOperands()) {
      Attribute cst;
      if (matchPattern(operand, m_Constant(&cst)))
        continue;
      auto def = operand.getDefiningOp<arith::SelectOp>();
      if (sel || !def)
        return failure();
      sel = def;
    }
    if (!sel || !sel.getCondition().getType().isInteger(1))
      return failure();
    Value arms[2] = {sel.getTrueValue(), sel.getFalseValue()};
    for (Value arm : arms) {
      Attribute cst;
      if (!matchPattern(arm, m_Constant(&cst)))
        return failure();
    }

    Value chosen[2];
    for (unsigned arm = 0; arm < 2; ++arm) {
      IRMapping map;
      map.map(sel.getResult(), arms[arm]);
      chosen[arm] = rewriter.clone(*op.getOperation(), map)->getResult(0);
    }
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, sel.getCondition(),
                                                 chosen[0], chosen[1]);
    return success();
  }
};

/// The if counterpart of SelectOfSameBaseGEPs: arms that index one base
/// differently choose the index instead, with a constant index materialized
/// where the gep kept it in an attribute.
template <typename IfT> struct IfOfSameBaseGEPs : public OpRewritePattern<IfT> {
  using OpRewritePattern<IfT>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfT ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp->getNumResults() == 0 || ifOp.getElseRegion().empty())
      return failure();
    Operation *thenY = ifOp.getThenRegion().front().getTerminator();
    Operation *elseY = ifOp.getElseRegion().front().getTerminator();

    SmallVector<Type> newTypes;
    SmallVector<LLVM::GEPOp> tmpl(ifOp->getNumResults(), nullptr);
    SmallVector<LLVM::GEPOp> other(ifOp->getNumResults(), nullptr);
    SmallVector<char> inBytes(ifOp->getNumResults(), 0);
    // Which arm, if any, yields the base itself: that arm stepped zero
    // elements of the other's type.
    SmallVector<int> bare(ifOp->getNumResults(), -1);
    bool any = false;
    for (auto [i, res] : llvm::enumerate(ifOp->getResults())) {
      newTypes.push_back(res.getType());
      auto t = dyn_cast_if_present<LLVM::GEPOp>(
          thenY->getOperand(i).getDefiningOp());
      auto f = dyn_cast_if_present<LLVM::GEPOp>(
          elseY->getOperand(i).getDefiningOp());
      if (!t && f && thenY->getOperand(i) == f.getBase()) {
        t = f;
        bare[i] = 0;
      } else if (!f && t && elseY->getOperand(i) == t.getBase()) {
        f = t;
        bare[i] = 1;
      }
      if (!t || !f || t.getBase() != f.getBase() ||
          t.getType() != f.getType() || t.getIndices().size() != 1 ||
          f.getIndices().size() != 1)
        continue;
      auto dynT = dyn_cast_if_present<Value>(t.getIndices()[0]);
      auto dynF = dyn_cast_if_present<Value>(f.getIndices()[0]);
      // Element types that disagree share no index scale, so the arms choose
      // a byte offset and the rebuilt gep walks bytes.
      bool bytes = t.getElemType() != f.getElemType();
      Type i64 = rewriter.getI64Type();
      if (bytes) {
        if ((dynT && dynT.getType() != i64) || (dynF && dynF.getType() != i64))
          continue;
      } else if (dynT && dynF && dynT.getType() != dynF.getType()) {
        continue;
      }
      tmpl[i] = t;
      other[i] = f;
      inBytes[i] = bytes;
      newTypes[i] = bytes  ? i64
                    : dynT ? dynT.getType()
                    : dynF ? dynF.getType()
                           : i64;
      any = true;
    }
    if (!any)
      return failure();

    auto newIf = createLikeIf(rewriter, ifOp, newTypes);
    for (unsigned r = 0; r < 2; ++r) {
      Region &from = r ? ifOp.getElseRegion() : ifOp.getThenRegion();
      Region &to = r ? newIf.getElseRegion() : newIf.getThenRegion();
      rewriter.inlineRegionBefore(from, to, to.begin());
      rewriter.eraseBlock(&to.back());
      Operation *y = to.front().getTerminator();
      SmallVector<Value> ops(y->getOperands());
      for (auto [i, g] : llvm::enumerate(tmpl)) {
        if (!g)
          continue;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(y);
        if (bare[i] == (int)r) {
          ops[i] =
              arith::ConstantOp::create(rewriter, ifOp.getLoc(), newTypes[i],
                                        rewriter.getIntegerAttr(newTypes[i], 0))
                  .getResult();
          continue;
        }
        auto gep = cast<LLVM::GEPOp>(ops[i].getDefiningOp());
        auto idx = gep.getIndices()[0];
        int64_t scale = inBytes[i]
                            ? (int64_t)DataLayout::closest(gep).getTypeSize(
                                  gep.getElemType())
                            : 1;
        if (auto dv = dyn_cast_if_present<Value>(idx)) {
          if (scale != 1) {
            Value k = arith::ConstantOp::create(
                          rewriter, ifOp.getLoc(), newTypes[i],
                          rewriter.getIntegerAttr(newTypes[i], scale))
                          .getResult();
            dv = arith::MulIOp::create(rewriter, ifOp.getLoc(), dv, k)
                     .getResult();
          }
          ops[i] = dv;
          continue;
        }
        ops[i] = arith::ConstantOp::create(
                     rewriter, ifOp.getLoc(), newTypes[i],
                     rewriter.getIntegerAttr(
                         newTypes[i], cast<IntegerAttr>(idx).getInt() * scale))
                     .getResult();
      }
      rewriter.modifyOpInPlace(y, [&] { y->setOperands(ops); });
    }

    rewriter.setInsertionPointAfter(newIf);
    SmallVector<Value> results;
    for (auto [i, g] : llvm::enumerate(tmpl)) {
      Value v = newIf->getResult(i);
      if (g) {
        Type elem = inBytes[i] ? rewriter.getI8Type() : g.getElemType();
        // Neither arm's guarantees hold on the other's path.
        v = LLVM::GEPOp::create(rewriter, g.getLoc(), g.getType(), elem,
                                g.getBase(), SmallVector<LLVM::GEPArg>{v},
                                g.getNoWrapFlags() & other[i].getNoWrapFlags())
                .getResult();
      }
      results.push_back(v);
    }
    rewriter.replaceOp(ifOp, results);
    return success();
  }
};

/// The other way two geps disagree: identical indices off different bases,
/// as MFEM's shared memory slices are indexed. The branch chooses the base.
/// Operands the arms share are defined above the branch already, since the
/// sibling arm could not otherwise name them.
static bool gepsDifferOnlyInBase(LLVM::GEPOp t, LLVM::GEPOp f) {
  if (t.getElemType() != f.getElemType() || t.getType() != f.getType() ||
      t.getRawConstantIndices() != f.getRawConstantIndices() ||
      t.getBase() == f.getBase() ||
      t.getDynamicIndices().size() != f.getDynamicIndices().size())
    return false;
  for (auto [a, b] :
       llvm::zip_equal(t.getDynamicIndices(), f.getDynamicIndices()))
    if (a != b)
      return false;
  return true;
}

template <typename IfT>
struct IfOfDifferentBaseGEPs : public OpRewritePattern<IfT> {
  using OpRewritePattern<IfT>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfT ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp->getNumResults() == 0 || ifOp.getElseRegion().empty())
      return failure();
    Operation *thenY = ifOp.getThenRegion().front().getTerminator();
    Operation *elseY = ifOp.getElseRegion().front().getTerminator();

    SmallVector<Type> newTypes;
    SmallVector<LLVM::GEPOp> tmpl(ifOp->getNumResults(), nullptr);
    SmallVector<LLVM::GEPOp> other(ifOp->getNumResults(), nullptr);
    bool any = false;
    for (auto [i, res] : llvm::enumerate(ifOp->getResults())) {
      newTypes.push_back(res.getType());
      auto t = dyn_cast_if_present<LLVM::GEPOp>(
          thenY->getOperand(i).getDefiningOp());
      auto f = dyn_cast_if_present<LLVM::GEPOp>(
          elseY->getOperand(i).getDefiningOp());
      if (!t || !f || !gepsDifferOnlyInBase(t, f))
        continue;
      tmpl[i] = t;
      other[i] = f;
      newTypes[i] = t.getBase().getType();
      any = true;
    }
    if (!any)
      return failure();

    auto newIf = createLikeIf(rewriter, ifOp, newTypes);
    for (unsigned r = 0; r < 2; ++r) {
      Region &from = r ? ifOp.getElseRegion() : ifOp.getThenRegion();
      Region &to = r ? newIf.getElseRegion() : newIf.getThenRegion();
      rewriter.inlineRegionBefore(from, to, to.begin());
      rewriter.eraseBlock(&to.back());
      Operation *y = to.front().getTerminator();
      SmallVector<Value> ops(y->getOperands());
      for (auto [i, g] : llvm::enumerate(tmpl))
        if (g)
          ops[i] = cast<LLVM::GEPOp>(ops[i].getDefiningOp()).getBase();
      rewriter.modifyOpInPlace(y, [&] { y->setOperands(ops); });
    }

    rewriter.setInsertionPointAfter(newIf);
    SmallVector<Value> results;
    for (auto [i, g] : llvm::enumerate(tmpl)) {
      Value v = newIf->getResult(i);
      if (g) {
        auto cloned = cast<LLVM::GEPOp>(rewriter.clone(*g.getOperation()));
        cloned.setOperand(0, v);
        // Neither arm's guarantees hold on the other's path.
        cloned.setNoWrapFlags(g.getNoWrapFlags() & other[i].getNoWrapFlags());
        v = cloned.getResult();
      }
      results.push_back(v);
    }
    rewriter.replaceOp(ifOp, results);
    return success();
  }
};

/// The select twin. A pointer result means the condition is a scalar i1: a
/// shaped condition selects elementwise and would need a shaped result.
struct SelectOfDifferentBaseGEPs : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const override {
    if (!isa<LLVM::LLVMPointerType>(sel.getType()))
      return failure();
    auto t = sel.getTrueValue().getDefiningOp<LLVM::GEPOp>();
    auto f = sel.getFalseValue().getDefiningOp<LLVM::GEPOp>();
    if (!t || !f || !gepsDifferOnlyInBase(t, f))
      return failure();
    Value base =
        arith::SelectOp::create(rewriter, sel.getLoc(), sel.getCondition(),
                                t.getBase(), f.getBase())
            .getResult();
    auto cloned = cast<LLVM::GEPOp>(rewriter.clone(*t.getOperation()));
    cloned.setOperand(0, base);
    cloned.setNoWrapFlags(t.getNoWrapFlags() & f.getNoWrapFlags());
    rewriter.replaceOp(sel, cloned.getResult());
    return success();
  }
};

// Whether nothing can observe the address itself: every use either reads or
// writes the pointed-to memory, or carries the pointer somewhere that is in
// turn only dereferenced. A comparison, an escape into a call, or a store of
// the pointer as a value all answer no.
static bool onlyDereferenced(Value root) {
  SmallVector<Value> todo{root};
  SmallPtrSet<Value, 8> seen;
  while (!todo.empty()) {
    Value v = todo.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        if (use.get() != gep.getBase())
          return false;
        todo.push_back(gep.getResult());
      } else if (isa<LLVM::AddrSpaceCastOp, LLVM::BitcastOp>(user)) {
        todo.push_back(user->getResult(0));
      } else if (auto sel = dyn_cast<arith::SelectOp>(user)) {
        if (use.get() == sel.getCondition())
          return false;
        todo.push_back(sel.getResult());
      } else if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(user)) {
        todo.push_back(p2m.getResult());
      } else if (auto m2p = dyn_cast<enzymexla::Memref2PointerOp>(user)) {
        todo.push_back(m2p.getResult());
      } else if (isa<affine::AffineYieldOp, scf::YieldOp>(user)) {
        // Yielded out of a branch, the pointer becomes that branch's result.
        Operation *parent = user->getParentOp();
        if (!isa<affine::AffineIfOp, scf::IfOp>(parent))
          return false;
        todo.push_back(parent->getResult(use.getOperandNumber()));
      } else if (isa<LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(
                     user)) {
        // Reads the pointed-to memory, never the pointer.
      } else if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        if (use.get() == store.getValue())
          return false;
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
        if (use.get() == store.getValueToStore())
          return false;
      } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (use.get() == store.getValueToStore())
          return false;
      } else if (auto rmw = dyn_cast<LLVM::AtomicRMWOp>(user)) {
        if (use.get() != rmw.getPtr())
          return false;
      } else if (auto rmw = dyn_cast<memref::AtomicRMWOp>(user)) {
        if (use.get() != rmw.getMemref())
          return false;
      } else if (auto rmw = dyn_cast<enzyme::AtomicRMWOp>(user)) {
        if (use.get() != rmw.getMemref())
          return false;
      } else if (auto rmw = dyn_cast<enzyme::AffineAtomicRMWOp>(user)) {
        if (use.get() != rmw.getMemref())
          return false;
      } else if (isa<LLVM::MemsetOp, LLVM::MemsetInlineOp, LLVM::MemcpyOp,
                     LLVM::MemcpyInlineOp, LLVM::MemmoveOp>(user)) {
        // Fills or copies the pointed-to memory, through either end.
      } else {
        return false;
      }
    }
  }
  return true;
}

// A pointer that is only ever dereferenced cannot tell a null apart from any
// other address: the null arm can only fault. So a select of a null with a
// real pointer, whose result reaches nothing but loads and stores, is the
// real pointer.
struct SelectOfNullPointer : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const override {
    if (!isa<LLVM::LLVMPointerType>(sel.getType()))
      return failure();
    bool trueNull = sel.getTrueValue().getDefiningOp<LLVM::ZeroOp>() != nullptr;
    bool falseNull =
        sel.getFalseValue().getDefiningOp<LLVM::ZeroOp>() != nullptr;
    if (trueNull == falseNull)
      return failure();
    if (!onlyDereferenced(sel.getResult()))
      return failure();
    rewriter.replaceOp(sel,
                       trueNull ? sel.getFalseValue() : sel.getTrueValue());
    return success();
  }
};

// The same fold where the guard is a branch rather than a select. The kept
// arm has to be defined outside the branch: a value computed inside it is not
// available where the result is used.
template <typename IfT> struct IfOfNullPointer : public OpRewritePattern<IfT> {
  using OpRewritePattern<IfT>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfT ifOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = ifOp;
    if (op->getNumResults() == 0 || op->getNumRegions() != 2 ||
        op->getRegion(0).empty() || op->getRegion(1).empty())
      return failure();
    Operation *thenTerm = op->getRegion(0).front().getTerminator();
    Operation *elseTerm = op->getRegion(1).front().getTerminator();
    if (thenTerm->getNumOperands() != op->getNumResults() ||
        elseTerm->getNumOperands() != op->getNumResults())
      return failure();

    auto definedOutside = [&](Value v) {
      if (Operation *def = v.getDefiningOp())
        return !op->isAncestor(def);
      return !op->isAncestor(cast<BlockArgument>(v).getOwner()->getParentOp());
    };

    bool changed = false;
    for (auto [i, res] : llvm::enumerate(op->getResults())) {
      if (!isa<LLVM::LLVMPointerType>(res.getType()) || res.use_empty())
        continue;
      Value tv = thenTerm->getOperand(i), fv = elseTerm->getOperand(i);
      bool trueNull = tv.getDefiningOp<LLVM::ZeroOp>() != nullptr;
      bool falseNull = fv.getDefiningOp<LLVM::ZeroOp>() != nullptr;
      if (trueNull == falseNull)
        continue;
      Value keep = trueNull ? fv : tv;
      if (!definedOutside(keep) || !onlyDereferenced(res))
        continue;
      rewriter.replaceAllUsesWith(res, keep);
      changed = true;
    }
    if (!changed)
      return failure();
    if (wouldOpBeTriviallyDead(op))
      rewriter.eraseOp(op);
    return success();
  }
};

// How many scalars an LLVM aggregate holds when every leaf is the same scalar
// type, or -1.
static int64_t homogeneousLeafCount(Type t, Type &leaf) {
  if (auto at = dyn_cast<LLVM::LLVMArrayType>(t)) {
    int64_t n = homogeneousLeafCount(at.getElementType(), leaf);
    return n < 0 ? -1 : n * at.getNumElements();
  }
  if (auto st = dyn_cast<LLVM::LLVMStructType>(t)) {
    if (st.isOpaque())
      return -1;
    int64_t tot = 0;
    for (Type f : st.getBody()) {
      int64_t n = homogeneousLeafCount(f, leaf);
      if (n < 0)
        return -1;
      tot += n;
    }
    return tot;
  }
  if (t.isIntOrFloat()) {
    if (!leaf)
      leaf = t;
    return leaf == t ? 1 : -1;
  }
  return -1;
}

// Scratch declared as one aggregate value (a union wrapping a register array)
// reaches the memref world as a memref of an LLVM struct whose only consumers
// cast it straight back to a pointer. Padding-free and single-leaf-typed, that
// is flat scalar scratch, and the pointer round trip then folds to a view.
struct FlattenAggregateAlloca : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    MemRefType MT = alloca.getType();
    if (!isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(MT.getElementType()) ||
        !MT.hasStaticShape() || !MT.getLayout().isIdentity())
      return failure();
    Type leaf;
    int64_t leaves = homogeneousLeafCount(MT.getElementType(), leaf);
    if (leaves <= 0)
      return failure();
    DataLayout dl = DataLayout::closest(alloca);
    if (dl.getTypeSize(MT.getElementType()) != leaves * dl.getTypeSize(leaf))
      return failure();
    if (!llvm::all_of(alloca->getUsers(),
                      llvm::IsaPred<enzymexla::Memref2PointerOp>))
      return failure();

    auto NT = MemRefType::get({leaves * MT.getNumElements()}, leaf,
                              MemRefLayoutAttrInterface{}, MT.getMemorySpace());
    auto flat = memref::AllocaOp::create(rewriter, alloca.getLoc(), NT,
                                         alloca.getAlignmentAttr());
    for (Operation *user : llvm::make_early_inc_range(alloca->getUsers())) {
      rewriter.setInsertionPoint(user);
      rewriter.replaceOpWithNewOp<enzymexla::Memref2PointerOp>(
          user, user->getResult(0).getType(), flat);
    }
    rewriter.eraseOp(alloca);
    return success();
  }
};

// A store of undef or poison leaves the memory holding any value, and what
// it held before is one of those, so the store does nothing.
struct StoreOfUndef
    : public OpInterfaceRewritePattern<enzyme::StoreLikeInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(enzyme::StoreLikeInterface store,
                                PatternRewriter &rewriter) const override {
    Operation *value = store.getStoredValue().getDefiningOp();
    if (!isa_and_nonnull<LLVM::UndefOp, LLVM::PoisonOp, ub::PoisonOp>(value))
      return failure();
    rewriter.eraseOp(store);
    return success();
  }
};

// A while loop whose exit test cannot come out differently on a third
// evaluation runs its before region at most twice and its after region at
// most once, so it unrolls to
//   before(inits); if (condition) { after; before(yields) }
// The test is decided by then when it is a pure function of the loop-carried
// arguments and each of those is yielded a loop-invariant value: the second
// evaluation already sees what every later one would. It is also decided when
// the bits known of the values feeding the second evaluation settle it, as
// for a trip flag yielded as a constant or a counter whose low bit it reads.
struct UnrollInvariantYieldWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Block &before = whileOp.getBefore().front();
    Block &after = whileOp.getAfter().front();
    auto conditionOp = cast<scf::ConditionOp>(before.getTerminator());
    auto yieldOp = cast<scf::YieldOp>(after.getTerminator());

    // The loop-carried arguments a value is a pure function of, together
    // with values from outside the loop; nullopt when the dependence passes
    // through a region or a side effect. An after-region argument stands for
    // the condition operand forwarded to it.
    auto carriedArguments =
        [&](Value root) -> std::optional<SetVector<BlockArgument>> {
      SetVector<BlockArgument> carried;
      BackwardSliceOptions options;
      options.inclusive = true;
      options.omitBlockArguments = true;
      options.filter = [&](Operation *op) { return whileOp->isAncestor(op); };
      SmallVector<Value> roots{root};
      DenseSet<Value> visited;
      while (!roots.empty()) {
        Value value = roots.pop_back_val();
        if (!visited.insert(value).second)
          continue;
        SmallVector<Value> operands{value};
        if (Operation *definingOp = value.getDefiningOp();
            definingOp && whileOp->isAncestor(definingOp)) {
          SetVector<Operation *> slice;
          if (failed(getBackwardSlice(definingOp, &slice, options)))
            return std::nullopt;
          for (Operation *op : slice) {
            if (op->getNumRegions() || !isMemoryEffectFree(op))
              return std::nullopt;
            operands.append(op->operand_begin(), op->operand_end());
          }
        }
        for (Value operand : operands) {
          auto blockArg = dyn_cast<BlockArgument>(operand);
          if (!blockArg)
            continue;
          Block *owner = blockArg.getOwner();
          if (owner == &before)
            carried.insert(blockArg);
          else if (owner == &after)
            roots.push_back(conditionOp.getArgs()[blockArg.getArgNumber()]);
          else if (whileOp->isAncestor(owner->getParentOp()))
            return std::nullopt;
        }
      }
      return carried;
    };
    auto isInvariant = [&](Value value) {
      auto carried = carriedArguments(value);
      return carried && carried->empty();
    };

    // The bits known of an integer value in one evaluation of the loop:
    // evaluation 0 is outside the loop, 1 the first evaluation of the before
    // region (fed by the inits), 2 the second (fed by the first's yields).
    // nullopt for a value that is not an integer.
    using Known = std::optional<KnownBits>;
    using Key = std::pair<Value, unsigned>;
    DenseMap<Key, Known> known;
    auto widthOf = [](Type type) -> unsigned {
      if (auto intType = dyn_cast<IntegerType>(type))
        return intType.getWidth();
      if (isa<IndexType>(type))
        return IndexType::kInternalStorageBitWidth;
      return 0;
    };
    auto keyOf = [&](Value value, unsigned evaluation) -> Key {
      auto blockArg = dyn_cast<BlockArgument>(value);
      Operation *definingOp = value.getDefiningOp();
      if (!whileOp->isAncestor(blockArg ? blockArg.getOwner()->getParentOp()
                                        : definingOp))
        evaluation = 0;
      return {value, evaluation};
    };
    // The values, with their evaluations, the bits of a key inside the loop
    // follow from.
    auto operandKeys = [&](Key key) -> SmallVector<Key> {
      auto [value, evaluation] = key;
      assert(evaluation != 0);
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        unsigned argNumber = blockArg.getArgNumber();
        if (blockArg.getOwner() == &before) {
          assert(evaluation == 1 || evaluation == 2);
          if (evaluation == 1)
            return {keyOf(whileOp.getInits()[argNumber], 0)};
          return {keyOf(yieldOp.getOperand(argNumber), 1)};
        }
        assert(blockArg.getOwner() == &after);
        return {keyOf(conditionOp.getArgs()[argNumber], evaluation)};
      }
      Operation *definingOp = value.getDefiningOp();
      SmallVector<Key> keys;
      if (auto ifOp = dyn_cast<scf::IfOp>(definingOp)) {
        unsigned resultNumber = cast<OpResult>(value).getResultNumber();
        keys.push_back(keyOf(ifOp.getCondition(), evaluation));
        keys.push_back(
            keyOf(ifOp.thenYield().getOperand(resultNumber), evaluation));
        keys.push_back(
            keyOf(ifOp.elseYield().getOperand(resultNumber), evaluation));
        return keys;
      }
      for (Value operand : definingOp->getOperands())
        keys.push_back(keyOf(operand, evaluation));
      return keys;
    };
    auto compute = [&](Key key, ArrayRef<Known> operands) -> Known {
      auto [value, evaluation] = key;
      unsigned width = widthOf(value.getType());
      if (isa<BlockArgument>(value))
        return operands[0];
      APInt constant;
      if (matchPattern(value, m_ConstantInt(&constant)))
        return KnownBits::makeConstant(constant);
      if (llvm::any_of(operands, [](const Known &bits) { return !bits; }))
        return KnownBits(width);
      Operation *definingOp = value.getDefiningOp();
      if (isa<scf::IfOp, arith::SelectOp>(definingOp)) {
        const KnownBits &condition = *operands[0];
        if (condition.isConstant())
          return operands[condition.isZero() ? 2 : 1];
        return operands[1]->intersectWith(*operands[2]);
      }
      if (auto cmp = dyn_cast<arith::CmpIOp>(definingOp)) {
        const KnownBits &lhs = *operands[0], &rhs = *operands[1];
        std::optional<bool> result;
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          result = KnownBits::eq(lhs, rhs);
          break;
        case arith::CmpIPredicate::ne:
          result = KnownBits::ne(lhs, rhs);
          break;
        case arith::CmpIPredicate::slt:
          result = KnownBits::slt(lhs, rhs);
          break;
        case arith::CmpIPredicate::sle:
          result = KnownBits::sle(lhs, rhs);
          break;
        case arith::CmpIPredicate::sgt:
          result = KnownBits::sgt(lhs, rhs);
          break;
        case arith::CmpIPredicate::sge:
          result = KnownBits::sge(lhs, rhs);
          break;
        case arith::CmpIPredicate::ult:
          result = KnownBits::ult(lhs, rhs);
          break;
        case arith::CmpIPredicate::ule:
          result = KnownBits::ule(lhs, rhs);
          break;
        case arith::CmpIPredicate::ugt:
          result = KnownBits::ugt(lhs, rhs);
          break;
        case arith::CmpIPredicate::uge:
          result = KnownBits::uge(lhs, rhs);
          break;
        }
        if (result)
          return KnownBits::makeConstant(APInt(1, *result));
        return KnownBits(width);
      }
      if (operands.size() == 2) {
        const KnownBits &lhs = *operands[0], &rhs = *operands[1];
        if (isa<arith::AddIOp>(definingOp))
          return KnownBits::add(lhs, rhs);
        if (isa<arith::SubIOp>(definingOp))
          return KnownBits::sub(lhs, rhs);
        if (isa<arith::MulIOp>(definingOp))
          return KnownBits::mul(lhs, rhs);
        if (isa<arith::AndIOp>(definingOp))
          return lhs & rhs;
        if (isa<arith::OrIOp>(definingOp))
          return lhs | rhs;
        if (isa<arith::XOrIOp>(definingOp))
          return lhs ^ rhs;
        if (isa<arith::ShLIOp>(definingOp))
          return KnownBits::shl(lhs, rhs);
        if (isa<arith::ShRUIOp>(definingOp))
          return KnownBits::lshr(lhs, rhs);
        if (isa<arith::ShRSIOp>(definingOp))
          return KnownBits::ashr(lhs, rhs);
        if (isa<arith::DivUIOp>(definingOp))
          return KnownBits::udiv(lhs, rhs);
        if (isa<arith::DivSIOp>(definingOp))
          return KnownBits::sdiv(lhs, rhs);
        if (isa<arith::RemUIOp>(definingOp))
          return KnownBits::urem(lhs, rhs);
        if (isa<arith::RemSIOp>(definingOp))
          return KnownBits::srem(lhs, rhs);
      }
      if (operands.size() == 1) {
        const KnownBits &operand = *operands[0];
        if (isa<arith::ExtUIOp, arith::IndexCastUIOp>(definingOp))
          return operand.zextOrTrunc(width);
        if (isa<arith::ExtSIOp, arith::IndexCastOp>(definingOp))
          return operand.sextOrTrunc(width);
        if (isa<arith::TruncIOp>(definingOp))
          return operand.trunc(width);
      }
      return KnownBits(width);
    };
    auto knownIn = [&](Value value, unsigned evaluation) -> Known {
      Key root = keyOf(value, evaluation);
      SmallVector<Key> worklist{root};
      DenseSet<Key> pending;
      while (!worklist.empty()) {
        Key key = worklist.back();
        if (known.contains(key)) {
          worklist.pop_back();
          continue;
        }
        unsigned width = widthOf(key.first.getType());
        if (!width) {
          known[key] = std::nullopt;
          worklist.pop_back();
          continue;
        }
        // Of a value from outside the loop only a constant says anything.
        if (key.second == 0) {
          APInt constant;
          known[key] = matchPattern(key.first, m_ConstantInt(&constant))
                           ? KnownBits::makeConstant(constant)
                           : KnownBits(width);
          worklist.pop_back();
          continue;
        }
        SmallVector<Key> operands = operandKeys(key);
        SmallVector<Key> missing;
        for (const Key &operand : operands)
          if (!known.contains(operand))
            missing.push_back(operand);
        if (missing.empty()) {
          SmallVector<Known> bits;
          for (const Key &operand : operands)
            bits.push_back(known[operand]);
          known[key] = compute(key, bits);
          pending.erase(key);
          worklist.pop_back();
          continue;
        }
        // Asked again before its operands settle, a value is on a cycle
        // within one evaluation; nothing is known of it.
        if (!pending.insert(key).second) {
          known[key] = KnownBits(width);
          worklist.pop_back();
          continue;
        }
        worklist.append(missing);
      }
      return known[root];
    };

    bool decided = false;
    if (auto carried = carriedArguments(conditionOp.getCondition()))
      decided = llvm::all_of(*carried, [&](BlockArgument blockArg) {
        return isInvariant(yieldOp.getOperand(blockArg.getArgNumber()));
      });
    if (!decided) {
      Known condition = knownIn(conditionOp.getCondition(), 2);
      decided = condition && condition->isZero();
    }
    if (!decided)
      return failure();

    Location loc = whileOp.getLoc();
    IRMapping firstEvaluation;
    for (auto [blockArg, init] :
         llvm::zip(before.getArguments(), whileOp.getInits()))
      firstEvaluation.map(blockArg, init);
    for (Operation &op : before.without_terminator())
      rewriter.clone(op, firstEvaluation);
    Value firstCondition =
        firstEvaluation.lookupOrDefault(conditionOp.getCondition());
    SmallVector<Value> firstForwarded;
    for (Value value : conditionOp.getArgs())
      firstForwarded.push_back(firstEvaluation.lookupOrDefault(value));

    auto ifOp = scf::IfOp::create(
        rewriter, loc, firstCondition,
        [&](OpBuilder &builder, Location loc) {
          IRMapping secondEvaluation;
          for (auto [afterArg, forwarded] :
               llvm::zip(after.getArguments(), firstForwarded))
            secondEvaluation.map(afterArg, forwarded);
          for (Operation &op : after.without_terminator())
            builder.clone(op, secondEvaluation);
          for (auto [blockArg, yielded] :
               llvm::zip(before.getArguments(), yieldOp.getOperands()))
            secondEvaluation.map(blockArg,
                                 secondEvaluation.lookupOrDefault(
                                     firstEvaluation.lookupOrDefault(yielded)));
          for (Operation &op : before.without_terminator())
            builder.clone(op, secondEvaluation);
          SmallVector<Value> secondForwarded;
          for (Value value : conditionOp.getArgs())
            secondForwarded.push_back(secondEvaluation.lookupOrDefault(value));
          scf::YieldOp::create(builder, loc, secondForwarded);
        },
        [&](OpBuilder &builder, Location loc) {
          scf::YieldOp::create(builder, loc, firstForwarded);
        });
    rewriter.replaceOp(whileOp, ifOp.getResults());
    return success();
  }
};

struct CanonicalizeParallelPass
    : public enzyme::impl::CanonicalizeParallelPassBase<
          CanonicalizeParallelPass> {
  using CanonicalizeParallelPassBase::CanonicalizeParallelPassBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();

    // The same pattern collection the canonicalizer performs.
    RewritePatternSet owningPatterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, ctx);
    owningPatterns.add<
        TruncOrConst, SelectOfSameBaseGEPs, SinkAddrSpaceCastThroughGEP,
        Pointer2MemrefOfAddrSpaceCast, IfOfSameBaseGEPs<scf::IfOp>,
        IfOfSameBaseGEPs<affine::AffineIfOp>, IfOfDifferentBaseGEPs<scf::IfOp>,
        IfOfDifferentBaseGEPs<affine::AffineIfOp>, SelectOfDifferentBaseGEPs,
        SinkThroughIfOfConstants<arith::IndexCastOp, scf::IfOp>,
        SinkThroughIfOfConstants<arith::IndexCastOp, affine::AffineIfOp>,
        SinkThroughIfOfConstants<arith::DivSIOp, scf::IfOp>,
        SinkThroughIfOfConstants<arith::DivSIOp, affine::AffineIfOp>,
        SinkThroughIfOfConstants<arith::DivUIOp, scf::IfOp>,
        SinkThroughIfOfConstants<arith::DivUIOp, affine::AffineIfOp>,
        SinkThroughIfOfConstants<arith::AddIOp, scf::IfOp>,
        SinkThroughIfOfConstants<arith::AddIOp, affine::AffineIfOp>,
        SinkThroughIfOfConstants<arith::MulIOp, scf::IfOp>,
        SinkThroughIfOfConstants<arith::MulIOp, affine::AffineIfOp>,
        SinkThroughSelectOfConstants<arith::IndexCastOp>,
        SinkThroughSelectOfConstants<arith::IndexCastUIOp>,
        SinkThroughSelectOfConstants<arith::DivSIOp>,
        SinkThroughSelectOfConstants<arith::DivUIOp>,
        SinkThroughSelectOfConstants<arith::AddIOp>,
        SinkThroughSelectOfConstants<arith::MulIOp>, SelectOfNullPointer,
        IfOfNullPointer<scf::IfOp>, IfOfNullPointer<affine::AffineIfOp>,
        FlattenAggregateAlloca, StoreOfUndef, UnrollInvariantYieldWhile>(ctx);
    FrozenRewritePatternSet patterns(std::move(owningPatterns));

    GreedyRewriteConfig config;
    config.enableFolding();
    config.enableConstantCSE();
    // The canonicalizer's default: no identical-block merging. Merging adds
    // successor operands for the values the blocks differed in, and e.g.
    // llvm.invoke cannot carry an index-typed successor operand.
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    // Convergence failure below matches the plain canonicalizer's behavior
    // of leaving the IR in its best-effort state.

    // Anchored on a function (the Enzyme postpasses run this over a single
    // freshly differentiated function) -- or on anything else that is not a
    // single-block symbol table -- the anchor itself is the unit.
    if (isa<FunctionOpInterface>(root) ||
        !root->hasTrait<OpTrait::SymbolTable>() || !root->getNumRegions() ||
        !root->getRegion(0).hasOneBlock()) {
      (void)applyPatternsGreedily(root, patterns, config);
      return;
    }

    SmallVector<Operation *> targets;
    SmallVector<Operation *> loose;
    for (Operation &op : root->getRegion(0).front()) {
      if (op.getNumRegions() != 0 &&
          llvm::any_of(op.getRegions(), [](Region &r) { return !r.empty(); }))
        targets.push_back(&op);
      else
        loose.push_back(&op);
    }

    if (!loose.empty()) {
      GreedyRewriteConfig looseConfig = config;
      looseConfig.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      (void)applyOpPatternsGreedily(loose, patterns, looseConfig);
    }

    if (parallel) {
      parallelForEach(ctx, targets, [&](Operation *op) {
        (void)applyPatternsGreedily(op, patterns, config);
      });
    } else {
      for (Operation *op : targets)
        (void)applyPatternsGreedily(op, patterns, config);
    }
  }
};

} // namespace
