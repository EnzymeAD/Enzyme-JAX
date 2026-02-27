#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_INJECTCAPABILITYPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct InjectCapabilityPass
    : public enzyme::pact::impl::InjectCapabilityPassBase<
          InjectCapabilityPass> {
  using InjectCapabilityPassBase::InjectCapabilityPassBase;

  void runOnOperation() override {
    // TODO: Inject target capability attributes onto PACT operations
  }
};

} // namespace
