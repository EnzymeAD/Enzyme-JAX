#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_SCHEMEALIGNMENTCHECKPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct SchemeAlignmentCheckPass
    : public enzyme::pact::impl::SchemeAlignmentCheckPassBase<
          SchemeAlignmentCheckPass> {
  using SchemeAlignmentCheckPassBase::SchemeAlignmentCheckPassBase;

  void runOnOperation() override {
    // TODO: Validate contract/capability key alignment against the scheme
  }
};

} // namespace
