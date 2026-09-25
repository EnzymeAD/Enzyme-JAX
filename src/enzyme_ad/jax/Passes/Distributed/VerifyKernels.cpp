#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DISTRIBUTEDVERIFYKERNELSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

// The per-op invariants (slot disjointness, per-tensor sharding-axis
// uniqueness) live in the op verifiers. Calling them here makes them hold at
// the entry of search and the exit of the frontend regardless of whether the
// surrounding PassManager or the CLI has the verifier enabled.
LogicalResult verifyDistributedOps(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    if (isa<DistributedKernelOp, DistributedFunctionOp, DistributedCallOp>(
            op) &&
        failed(mlir::verify(op, /*verifyRecursively=*/false)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

namespace {
struct DistributedVerifyKernelsPass
    : public impl::DistributedVerifyKernelsPassBase<
          DistributedVerifyKernelsPass> {
  using DistributedVerifyKernelsPassBase::DistributedVerifyKernelsPassBase;

  void runOnOperation() override {
    if (failed(verifyDistributedOps(getOperation())))
      signalPassFailure();
  }
};
} // namespace

} // namespace mlir::enzyme::distributed
