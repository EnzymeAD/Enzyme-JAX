// TODO: replace meet with CapabilityRegistry-driven evaluation.
// TODO: introduce EvalContext for derived properties and CostHelper for
// strategy ranking.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_ADAPTIVEMEETPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct AdaptiveMeetPass
    : public enzyme::pact::impl::AdaptiveMeetPassBase<AdaptiveMeetPass> {
  using AdaptiveMeetPassBase::AdaptiveMeetPassBase;

  void runOnOperation() override {
    // TODO: Reconcile contracts and capabilities, produce pact.plan
  }
};

} // namespace
