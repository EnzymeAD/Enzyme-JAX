//===- SplitBranchedAccesses.cpp - Accesses at a chosen index ------------===//
//
// An access whose index is chosen by a branch reads or writes a place the
// forwarding cannot name: the index is not a constant, so it may be any slot
// of the allocation and blocks the one it reads. Where the branch chooses
// between constants, the access can be done in each arm instead, at the
// constant that arm chose, which is a place the forwarding does know.
//
// The arithmetic that stood between a branch and an access -- a byte offset
// cast and scaled into an element index -- is sunk into the arms beforehand
// by canonicalization, so the index arrives here as the branch's own result.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SPLITBRANCHEDACCESSESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

/// The branch an index came from.
struct BranchedIndex {
  Operation *ifOp;
  unsigned resultNo;
};

static bool yieldsConstant(Operation *yield, unsigned resultNo) {
  Attribute cst;
  return matchPattern(yield->getOperand(resultNo), m_Constant(&cst));
}

/// Both arms of an if hand back a constant for this result. An if without an
/// else has an empty second region and chooses nothing.
static bool bothArmsConstant(Operation *ifOp, unsigned resultNo) {
  for (Region &arm : ifOp->getRegions()) {
    if (arm.empty())
      return false;
    if (!yieldsConstant(arm.front().getTerminator(), resultNo))
      return false;
  }
  return true;
}

/// The branch that chose an index, when the index is a branch's result and
/// both arms hand back a constant.
static std::optional<BranchedIndex> branchThatChose(Value index) {
  auto res = dyn_cast<OpResult>(index);
  if (!res)
    return std::nullopt;
  Operation *ifOp = res.getOwner();
  if (!isa<scf::IfOp, affine::AffineIfOp>(ifOp))
    return std::nullopt;
  if (!bothArmsConstant(ifOp, res.getResultNumber()))
    return std::nullopt;
  return BranchedIndex{ifOp, res.getResultNumber()};
}

static Operation *makeIfLike(OpBuilder &builder, Operation *ifOp,
                             TypeRange results) {
  if (auto sif = dyn_cast<scf::IfOp>(ifOp))
    return scf::IfOp::create(builder, sif.getLoc(), results, sif.getCondition(),
                             /*withElseRegion=*/true);
  auto aif = cast<affine::AffineIfOp>(ifOp);
  return affine::AffineIfOp::create(builder, aif.getLoc(), results,
                                    aif.getIntegerSet(), aif.getOperands(),
                                    /*withElseRegion=*/true);
}

/// Rebuilds `access` in each arm of a copy of `br.ifOp`, at the constant that
/// arm chose.
static void splitAccess(Operation *access, Value index,
                        const BranchedIndex &br) {
  // The branch is asked again where the access already stands, rather than
  // the access being carried up to where the branch was: everything the
  // access names is in hand here, and nothing has to be shown to survive a
  // move. What the branch is asked -- its condition, or the set and the
  // values it is taken over -- was in hand before it, so it is in hand here
  // too.
  OpBuilder builder(access);
  Operation *newIf = makeIfLike(builder, br.ifOp, access->getResultTypes());

  // The regions of both flavours of if are the arms, in order.
  for (unsigned arm = 0; arm < 2; ++arm) {
    Block *body = &newIf->getRegion(arm).front();
    OpBuilder armBuilder(body, body->begin());
    IRMapping map;
    // The constant this arm chose stands in for the branch's result.
    Value chose = br.ifOp->getRegion(arm).front().getTerminator()->getOperand(
        br.resultNo);
    Operation *chosen = armBuilder.clone(*chose.getDefiningOp());
    map.map(br.ifOp->getResult(br.resultNo), chosen->getResult(0));
    Operation *cloned = armBuilder.clone(*access, map);
    if (cloned->getNumResults()) {
      if (isa<scf::IfOp>(newIf))
        scf::YieldOp::create(armBuilder, access->getLoc(),
                             cloned->getResults());
      else
        affine::AffineYieldOp::create(armBuilder, access->getLoc(),
                                      cloned->getResults());
    }
  }

  access->replaceAllUsesWith(newIf->getResults());
  access->erase();
}

/// The index of an access, when it has exactly one.
static Value soleIndex(Operation *op) {
  if (auto ld = dyn_cast<memref::LoadOp>(op))
    return ld.getIndices().size() == 1 ? ld.getIndices()[0] : Value();
  if (auto st = dyn_cast<memref::StoreOp>(op))
    return st.getIndices().size() == 1 ? st.getIndices()[0] : Value();
  if (auto ld = dyn_cast<affine::AffineLoadOp>(op))
    return ld.getMapOperands().size() == 1 ? ld.getMapOperands()[0] : Value();
  if (auto st = dyn_cast<affine::AffineStoreOp>(op))
    return st.getMapOperands().size() == 1 ? st.getMapOperands()[0] : Value();
  return Value();
}

struct SplitBranchedAccessesPass
    : public enzyme::impl::SplitBranchedAccessesPassBase<
          SplitBranchedAccessesPass> {
  using SplitBranchedAccessesPassBase::SplitBranchedAccessesPassBase;

  void runOnOperation() override {
    SmallVector<std::pair<Operation *, BranchedIndex>> work;
    getOperation()->walk([&](Operation *op) {
      if (!isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
               affine::AffineStoreOp>(op))
        return;
      Value index = soleIndex(op);
      if (!index)
        return;
      if (auto br = branchThatChose(index))
        work.emplace_back(op, *br);
    });

    for (auto &[op, br] : work)
      splitAccess(op, soleIndex(op), br);
  }
};

} // namespace
