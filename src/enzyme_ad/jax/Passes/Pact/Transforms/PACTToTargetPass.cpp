#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_PACTTOTARGETPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct PACTToTargetPass
    : public enzyme::pact::impl::PACTToTargetPassBase<PACTToTargetPass> {
  using PACTToTargetPassBase::PACTToTargetPassBase;

  void runOnOperation() override {
    // TODO: Lower PACT ops guided by pact.plan
  }
};

} // namespace
