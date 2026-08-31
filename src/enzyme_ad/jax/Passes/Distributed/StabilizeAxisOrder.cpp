#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_STABILIZEAXISORDERPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Axis-algebra ops (distributed.LogicalMeshAxes, axis.factor, axis.product,
// axis.map, ...) carry no ordering semantics of their own: any legal
// placement within the (graph-region, no-dominance) module body is
// equally correct. In practice their textual position is decided
// incidentally by whichever pass happens to create them, which can depend
// on process-local, non-semantic details (e.g. hashing of internal symbol
// ids) that differ across separate compiler invocations processing the
// exact same input. That makes two equally-valid outputs of the same
// pipeline diff-incompatible with each other, which is purely a testing /
// reproducibility nuisance rather than a semantic bug.
//
// This function gives such ops a single canonical answer: it moves every
// MaybeTemporaryInterface op to sit immediately before its first use,
// walking the surrounding "real" (non-axis-algebra) ops in their existing,
// already-deterministic order. Because module bodies are graph regions,
// this reordering can never change semantics or break verification.
void stabilizeAxisOrder(ModuleOp moduleOp) {
  Block &block = moduleOp.getBodyRegion().front();

  // Snapshot before mutating, since we're about to detach/move ops in
  // `block` and the original op list would otherwise change under us.
  llvm::SmallVector<Operation *> snapshot;
  for (Operation &op : block) {
    snapshot.push_back(&op);
  }

  // Detach every axis-algebra op. The "real" ops left behind keep their
  // original relative order, since removing other ops can't reorder them.
  for (Operation *op : snapshot) {
    if (dyn_cast<axis::MaybeTemporaryInterface>(op)) {
      op->remove();
    }
  }

  // Re-append the real ops in their original order, materializing any
  // not-yet-manifested axis-algebra dependency chain immediately before
  // each one. This produces a full, deterministic rebuild of the block
  // driven only by the (already-deterministic) order of the real ops.
  //
  // Neither DistributedFunctionOp nor DistributedKernelOp is
  // IsolatedFromAbove, so ops nested arbitrarily deep inside a real op's
  // regions (e.g. a kernel body) may reference axis-algebra values
  // directly. `walk` recurses into nested regions in a fixed, deterministic
  // pre-order, so it's a convenient way to find every such reference
  // without hand-rolling the recursion here.
  for (Operation *op : snapshot) {
    if (dyn_cast<axis::MaybeTemporaryInterface>(op)) {
      continue; // Placed recursively below, or dead and cleaned up after.
    }
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        if (!definingOp) {
          continue;
        }
        if (auto maybeTemporary =
                dyn_cast<axis::MaybeTemporaryInterface>(definingOp);
            maybeTemporary && !maybeTemporary.isManifested()) {
          maybeTemporary.materialize(block);
        }
      }
    });
    op->moveBefore(&block, block.end());
  }

  // We exepect everything with a use to be inserted already:
  // fallback remark + erasure of anything unexpected.
  for (Operation *op : llvm::reverse(snapshot)) {
    if (auto maybeTemporary = dyn_cast<axis::MaybeTemporaryInterface>(op);
        maybeTemporary && !maybeTemporary.isManifested()) {
      op->emitRemark() << "unreferenced axis-algebra op survived to "
                          "stabilize-axis-order; expected --canonicalize to "
                          "have removed it";
      op->erase();
    }
  }
}

struct StabilizeAxisOrderPass
    : public impl::StabilizeAxisOrderPassBase<StabilizeAxisOrderPass> {
  using StabilizeAxisOrderPassBase::StabilizeAxisOrderPassBase;

  void runOnOperation() override { stabilizeAxisOrder(getOperation()); }
};

} // namespace

} // namespace mlir::enzyme::distributed
