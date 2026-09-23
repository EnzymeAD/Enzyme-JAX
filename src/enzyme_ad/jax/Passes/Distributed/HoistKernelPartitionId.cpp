#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/Builders.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_HOISTKERNELPARTITIONIDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// stablehlo.partition_id shows up inside a DistributedKernelOp body as a
// side effect of Shardy's own local lowering (see LowerKernels.cpp's comment
// on where it comes from), not something this dialect introduces itself. Its
// result is a rank-0 scalar, so hoisting it out costs nothing on the
// local/global scope invariant DistributedKernelOp::verify() enforces: local
// and global scope are the same type, and the new argument_shardings entry
// has no dimensions to declare a partitioning axis over. Downstream
// consumers that walk a kernel body as an opaque subgraph (ACT's own cost
// model, in particular, which has no notion of this op at all) then see only
// an ordinary scalar operand, never the op itself.
void hoistPartitionId(DistributedKernelOp kernelOp) {
  Block &body = kernelOp.getBody().front();
  SmallVector<stablehlo::PartitionIdOp> found;
  for (Operation &op : body.without_terminator()) {
    if (auto partitionId = dyn_cast<stablehlo::PartitionIdOp>(op)) {
      found.push_back(partitionId);
    }
  }
  if (found.empty()) {
    return;
  }

  // Every occurrence is deterministic and identical (no operands), so all of
  // them collapse onto one new hoisted value rather than one per occurrence.
  OpBuilder builder(kernelOp);
  Value hoisted = builder.create<stablehlo::PartitionIdOp>(kernelOp.getLoc());

  kernelOp.getArgumentsMutable().append(hoisted);
  BlockArgument newArg = body.addArgument(hoisted.getType(), kernelOp.getLoc());

  MLIRContext *ctx = kernelOp.getContext();
  SmallVector<IndexedTensorShardingAttr> shardings(
      kernelOp.getArgumentShardings().getShardings());
  shardings.push_back(buildEmptyShardingForType(ctx, hoisted.getType()));
  kernelOp.setArgumentShardingsAttr(
      IndexedTensorShardingPerValueAttr::get(ctx, shardings));

  for (stablehlo::PartitionIdOp partitionId : found) {
    partitionId.getResult().replaceAllUsesWith(newArg);
    partitionId.erase();
  }
}

struct HoistKernelPartitionIdPass
    : public impl::HoistKernelPartitionIdPassBase<HoistKernelPartitionIdPass> {
  using HoistKernelPartitionIdPassBase::HoistKernelPartitionIdPassBase;

  void runOnOperation() override {
    getOperation().walk(
        [](DistributedKernelOp kernelOp) { hoistPartitionId(kernelOp); });
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
