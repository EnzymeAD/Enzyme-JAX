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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZEPARALLELPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

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

    auto materialize = [&](LLVM::GEPOp gep, Value dyn) -> Value {
      if (dyn)
        return dyn;
      auto attr = cast<IntegerAttr>(gep.getIndices()[0]);
      Value c = arith::ConstantOp::create(
          rewriter, gep.getLoc(),
          IntegerAttr::get(idxTy, attr.getValue().getSExtValue()));
      return c;
    };
    Value idxT = materialize(gepT, dynT);
    Value idxF = materialize(gepF, dynF);
    Value idx = arith::SelectOp::create(rewriter, sel.getLoc(),
                                        sel.getCondition(), idxT, idxF);
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        sel, sel.getType(), gepT.getElemType(), gepT.getBase(),
        SmallVector<LLVM::GEPArg>{idx},
        gepT.getNoWrapFlags() & gepF.getNoWrapFlags());
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
    owningPatterns.add<TruncOrConst, SelectOfSameBaseGEPs>(ctx);
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
