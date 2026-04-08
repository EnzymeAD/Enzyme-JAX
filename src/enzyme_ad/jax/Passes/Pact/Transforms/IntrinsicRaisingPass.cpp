#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h"

namespace mlir {
namespace enzyme {
namespace pact {
#define GEN_PASS_DEF_INTRINSICRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"
} // namespace pact
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme::pact;

namespace {

struct IntrinsicRaisingPass
    : public enzyme::pact::impl::IntrinsicRaisingPassBase<
          IntrinsicRaisingPass> {
  using IntrinsicRaisingPassBase::IntrinsicRaisingPassBase;

  void runOnOperation() override {
    // TODO: Implement NVVM intrinsic raising patterns
  }
};

} // namespace
