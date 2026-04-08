#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_DISCHARGEPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct DischargePass
    : public enzyme::pact::impl::DischargePassBase<DischargePass> {
  using DischargePassBase::DischargePassBase;

  void runOnOperation() override {
    // TODO: Verify and discharge all PACT artifacts
  }
};

} // namespace
