//===- SICMGuardVersioning.cpp - Version code on a hoisted SICM guard -----===//

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SICMGUARDVERSIONINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// Must match kRuntimeGuardAttr in SampleInvariantCodeMotion.cpp.
constexpr llvm::StringLiteral kRuntimeGuardAttr = "sicm.runtime_guard";

static void foldKnownPredicate(IRRewriter &rewriter, Value known, bool value) {
  while (true) {
    Operation *user = nullptr;
    for (Operation *u : known.getUsers()) {
      auto select = dyn_cast<stablehlo::SelectOp>(u);
      auto ifOp = dyn_cast<stablehlo::IfOp>(u);
      if ((select && select.getPred() == known) ||
          (ifOp && ifOp.getPred() == known &&
           ifOp->hasAttr(kRuntimeGuardAttr))) {
        user = u;
        break;
      }
    }
    if (!user)
      return;
    if (auto select = dyn_cast<stablehlo::SelectOp>(user)) {
      rewriter.replaceOp(select,
                         value ? select.getOnTrue() : select.getOnFalse());
      continue;
    }
    auto ifOp = cast<stablehlo::IfOp>(user);
    Block *branch =
        value ? &ifOp.getTrueBranch().front() : &ifOp.getFalseBranch().front();
    Operation *term = branch->getTerminator();
    SmallVector<Value> results(term->getOperands());
    rewriter.eraseOp(term);
    rewriter.inlineBlockBefore(branch, ifOp);
    rewriter.replaceOp(ifOp, results);
  }
}

struct SICMGuardVersioningPass
    : public enzyme::impl::SICMGuardVersioningPassBase<
          SICMGuardVersioningPass> {
  using SICMGuardVersioningPassBase::SICMGuardVersioningPassBase;

  bool version(IRRewriter &rewriter, Value pred,
               ArrayRef<stablehlo::IfOp> guards) {
    Block *block = pred.getParentBlock();
    SetVector<Operation *> containers;
    for (stablehlo::IfOp guard : guards) {
      Operation *c = block->findAncestorOpInBlock(*guard);
      if (!c || c == guard.getOperation())
        return false;
      containers.insert(c);
    }
    Operation *last = containers.front();
    for (Operation *c : containers)
      if (last->isBeforeInBlock(c))
        last = c;

    Operation *def = pred.getDefiningOp();
    Block::iterator begin =
        def ? std::next(def->getIterator()) : block->begin();
    DenseSet<Value> tainted = {pred};
    SetVector<Operation *> moved;
    bool effectsMoved = false;
    for (Operation &op :
         llvm::make_range(begin, std::next(last->getIterator()))) {
      SetVector<Value> used(op.getOperands().begin(), op.getOperands().end());
      getUsedValuesDefinedAbove(op.getRegions(), used);
      bool dependent =
          containers.contains(&op) ||
          llvm::any_of(used, [&](Value v) { return tainted.contains(v); });
      if (!dependent) {
        // `op` stays before the moved ops: it must commute with them.
        if (effectsMoved && !isMemoryEffectFree(&op))
          return false;
        continue;
      }
      moved.insert(&op);
      tainted.insert(op.getResults().begin(), op.getResults().end());
      effectsMoved |= !isMemoryEffectFree(&op);
    }

    SmallVector<Value> escaping;
    for (Operation *op : moved)
      for (Value r : op->getResults())
        if (llvm::any_of(r.getUsers(), [&](Operation *user) {
              return !moved.contains(block->findAncestorOpInBlock(*user));
            }))
          escaping.push_back(r);
    SmallVector<Type> types;
    for (Value v : escaping)
      types.push_back(v.getType());

    Location loc = last->getLoc();
    rewriter.setInsertionPointAfter(last);
    auto ifOp = stablehlo::IfOp::create(rewriter, loc, types, pred);
    auto predType = RankedTensorType::get({}, rewriter.getI1Type());
    for (bool value : {true, false}) {
      Region &region = value ? ifOp.getTrueBranch() : ifOp.getFalseBranch();
      Block *arm = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(arm);
      Value known = stablehlo::ConstantOp::create(
          rewriter, loc, predType,
          DenseElementsAttr::get(predType, rewriter.getBoolAttr(value)));
      IRMapping mapping;
      mapping.map(pred, known);
      for (Operation *op : moved)
        rewriter.clone(*op, mapping);
      SmallVector<Value> yields;
      for (Value v : escaping)
        yields.push_back(mapping.lookup(v));
      stablehlo::ReturnOp::create(rewriter, loc, yields);
      foldKnownPredicate(rewriter, known, value);
      for (Operation *op = arm->getTerminator()->getPrevNode(); op;) {
        Operation *prev = op->getPrevNode();
        if (isOpTriviallyDead(op))
          rewriter.eraseOp(op);
        op = prev;
      }
    }
    for (auto [v, r] : llvm::zip(escaping, ifOp.getResults()))
      rewriter.replaceAllUsesWith(v, r);
    for (Operation *op : llvm::reverse(moved))
      rewriter.eraseOp(op);
    for (Operation *op = ifOp->getPrevNode(); op && op != def;) {
      Operation *prev = op->getPrevNode();
      if (isOpTriviallyDead(op))
        rewriter.eraseOp(op);
      op = prev;
    }
    return true;
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    // One predicate per round: versioning clones other predicates' guards.
    while (true) {
      Value pred;
      SmallVector<stablehlo::IfOp> guards;
      getOperation()->walk([&](stablehlo::IfOp op) {
        if (!op->hasAttr(kRuntimeGuardAttr))
          return;
        if (!pred)
          pred = op.getPred();
        if (op.getPred() == pred)
          guards.push_back(op);
      });
      if (!pred)
        return;
      if (!version(rewriter, pred, guards))
        for (stablehlo::IfOp guard : guards)
          guard->removeAttr(kRuntimeGuardAttr);
    }
  }
};

} // namespace
