#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DROPTRIVIALKERNELSHARDINGPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct DropTrivialKernelShardingPass
    : public impl::DropTrivialKernelShardingPassBase<
          DropTrivialKernelShardingPass> {
  using DropTrivialKernelShardingPassBase::DropTrivialKernelShardingPassBase;

  void runOnOperation() override {
    getOperation().walk([&](DistributedKernelOp kernelOp) {
      if (!isTriviallyLocalKernel(kernelOp)) {
        return;
      }
      MLIRContext *ctx = kernelOp.getContext();

      SmallVector<IndexedTensorShardingAttr> argumentShardings;
      argumentShardings.reserve(kernelOp.getArguments().size());
      for (Value operand : kernelOp.getArguments()) {
        argumentShardings.push_back(
            buildEmptyShardingForType(ctx, operand.getType()));
      }
      kernelOp.setArgumentShardingsAttr(
          IndexedTensorShardingPerValueAttr::get(ctx, argumentShardings));

      SmallVector<IndexedTensorShardingAttr> outputShardings;
      outputShardings.reserve(kernelOp.getResults().size());
      for (Value result : kernelOp.getResults()) {
        outputShardings.push_back(
            buildEmptyShardingForType(ctx, result.getType()));
      }
      kernelOp.setOutputShardingsAttr(
          IndexedTensorShardingPerValueAttr::get(ctx, outputShardings));

      kernelOp.getPartitioningAxesMutable().assign(ValueRange());
    });
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
