#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DROPKERNELBODYSHARDINGATTRSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct DropKernelBodyShardingAttrsPass
    : public impl::DropKernelBodyShardingAttrsPassBase<
          DropKernelBodyShardingAttrsPass> {
  using DropKernelBodyShardingAttrsPassBase::
      DropKernelBodyShardingAttrsPassBase;

  void runOnOperation() override {
    getOperation().walk([&](DistributedKernelOp kernelOp) {
      kernelOp.getBody().walk([&](Operation *op) {
        op->removeAttr("distributed.argument_shardings");
        op->removeAttr("distributed.output_shardings");
      });
    });
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
