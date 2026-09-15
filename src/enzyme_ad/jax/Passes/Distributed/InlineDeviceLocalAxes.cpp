#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_INLINEDEVICELOCALAXESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct InlineDeviceLocalAxesPass
    : public impl::InlineDeviceLocalAxesPassBase<InlineDeviceLocalAxesPass> {
  using InlineDeviceLocalAxesPassBase::InlineDeviceLocalAxesPassBase;

  void runOnOperation() override {
    // TODO: inline distributed.DeviceLocalAxis markers left behind by
    // LowerKernelsPass into concrete local structure.
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
