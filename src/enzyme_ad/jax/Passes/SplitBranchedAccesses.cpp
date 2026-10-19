//===- SplitBranchedAccesses.cpp - Accesses at a chosen index ------------===//
//
// An access whose index is chosen by a branch reads or writes a place the
// forwarding cannot name: the index is not a constant, so it may be any slot
// of the allocation and blocks the one it reads. Where the branch chooses
// between constants, the access can be done in each arm instead, at the
// constant that arm chose, which is a place the forwarding does know.
//
// The same holds for a branch choosing between values already in hand at the
// access, and for an index computed from the branch's result: done in each
// arm, the access is at an index the arm's own value gives, which the affine
// analyses can follow where the branch's result is opaque to them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
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

/// The branch an index came from, and the ops that computed the index from
/// its result, in order.
struct BranchedIndex {
  Operation *ifOp;
  unsigned resultNo;
  SmallVector<Operation *> chain;
};

/// What an arm chose can stand at the access: a constant, rebuilt in the arm,
/// or a value already in hand there.
static bool chosenInHand(Value chose, Operation *access, DominanceInfo &dom) {
  Attribute cst;
  return matchPattern(chose, m_Constant(&cst)) ||
         dom.properlyDominates(chose, access);
}

/// Both arms of an if hand back a value that can stand at the access. An if
/// without an else has an empty second region and chooses nothing.
static bool bothArmsInHand(Operation *ifOp, unsigned resultNo,
                           Operation *access, DominanceInfo &dom) {
  for (Region &arm : ifOp->getRegions()) {
    if (arm.empty())
      return false;
    if (!chosenInHand(arm.front().getTerminator()->getOperand(resultNo), access,
                      dom))
      return false;
  }
  return true;
}

/// Integer arithmetic an index is computed with, which an arm can redo.
static bool isIndexArith(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::IndexCastOp,
             arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::TruncIOp>(op);
}

/// The search for the branch an index came from.
struct BranchSearch {
  Operation *access;
  DominanceInfo &dom;
  std::optional<BranchedIndex> found;
  DenseMap<Value, bool> seen;
  bool conflict = false;
};

/// Whether `v` was computed from the result of a branch the access can be
/// split over, recording that branch and the index arithmetic in between.
static bool dependsOnBranch(Value v, unsigned depth, BranchSearch &search) {
  if (auto it = search.seen.find(v); it != search.seen.end())
    return it->second;
  bool depends = false;
  if (auto res = dyn_cast<OpResult>(v)) {
    Operation *def = res.getOwner();
    if (isa<scf::IfOp, affine::AffineIfOp>(def) &&
        bothArmsInHand(def, res.getResultNumber(), search.access, search.dom)) {
      if (!search.found)
        search.found = BranchedIndex{def, res.getResultNumber(), {}};
      else if (search.found->ifOp != def ||
               search.found->resultNo != res.getResultNumber())
        search.conflict = true;
      depends = true;
    } else if (depth < 8 && isIndexArith(def)) {
      for (Value operand : def->getOperands())
        depends |= dependsOnBranch(operand, depth + 1, search);
      if (depends)
        search.found->chain.push_back(def);
    }
  }
  search.seen[v] = depends;
  return depends;
}

/// The branch that chose an index: a result of an if, reached from the index
/// through index arithmetic, whose arms each hand back a value that can stand
/// at the access. The index arithmetic that does not depend on the branch
/// stays where it is. An index two branches chose is left alone.
static std::optional<BranchedIndex>
branchThatChose(Value index, Operation *access, DominanceInfo &dom) {
  BranchSearch search{access, dom, std::nullopt, {}};
  dependsOnBranch(index, 0, search);
  if (search.conflict)
    return std::nullopt;
  return search.found;
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

/// Rebuilds `access`, and the index arithmetic it was reached through, in
/// each arm of a copy of `br.ifOp`, at the value that arm chose.
static void splitAccess(Operation *access, const BranchedIndex &br) {
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
    // The value this arm chose stands in for the branch's result: a constant
    // is rebuilt in the arm, anything else is already in hand.
    Value chose = br.ifOp->getRegion(arm).front().getTerminator()->getOperand(
        br.resultNo);
    Attribute cst;
    if (matchPattern(chose, m_Constant(&cst)))
      chose = armBuilder.clone(*chose.getDefiningOp())->getResult(0);
    map.map(br.ifOp->getResult(br.resultNo), chose);
    for (Operation *op : br.chain)
      armBuilder.clone(*op, map);
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
    DominanceInfo dom(getOperation());
    SmallVector<std::pair<Operation *, BranchedIndex>> work;
    getOperation()->walk([&](Operation *op) {
      if (!isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
               affine::AffineStoreOp>(op))
        return;
      Value index = soleIndex(op);
      if (!index)
        return;
      if (auto br = branchThatChose(index, op, dom))
        work.emplace_back(op, *br);
    });

    for (auto &[op, br] : work)
      splitAccess(op, br);
  }
};

} // namespace
