#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_IDIOMRECOGNITIONPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct IdiomRecognitionPass
    : public enzyme::pact::impl::IdiomRecognitionPassBase<
          IdiomRecognitionPass> {
  using IdiomRecognitionPassBase::IdiomRecognitionPassBase;

  void runOnOperation() override {
    // TODO: Implement GPU idiom recognition (L4 -> L3)
  }
};

} // namespace
