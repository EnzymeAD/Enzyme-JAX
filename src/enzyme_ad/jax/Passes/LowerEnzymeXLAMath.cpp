// This must come first for windows builds
#define _USE_MATH_DEFINES

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla-math"

#include <functional>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMATHPASS
#define GEN_PASS_DEF_LOWERENZYMEXLAMLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

template <typename T>
static stablehlo::ConstantOp
createConstantOpFromScalar(PatternRewriter &rewriter, Location loc, Type type,
                           T value) {
  return stablehlo::ConstantOp::create(
      rewriter, loc, type,
      cast<ElementsAttr>(mlir::enzyme::makeAttr(type, value)));
}

namespace {
#include "src/enzyme_ad/jax/Passes/LowerEnzymeXLAMathPatterns.cpp.inc"

void lowerEnzymeXLAMath(Operation *op,
                        std::function<void()> signalPassFailure) {
  auto context = op->getContext();
  RewritePatternSet patterns(context);

  populateWithGenerated(patterns);

  GreedyRewriteConfig config;
  config.enableFolding();
  if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
    signalPassFailure();
  }

  // Verify that all illegal ops have been lowered
  auto walkResult = op->walk([&](Operation *local_op) {
    if (local_op->getName().getStringRef().starts_with("enzymexla.math.")) {
      local_op->emitError("Failed to lower enzymexla math operation");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    signalPassFailure();
  }
}
} // namespace

struct LowerEnzymeXLAMathPass
    : public enzyme::impl::LowerEnzymeXLAMathPassBase<LowerEnzymeXLAMathPass> {
  using Base::Base;

  void runOnOperation() override {
    lowerEnzymeXLAMath(getOperation(), [this]() { signalPassFailure(); });
  }
};

// TODO: delete this once Reactant uses `lower-enzymexla-math` instead of
// `lower-enzymexla-ml`
struct LowerEnzymeXLAMLPass
    : public enzyme::impl::LowerEnzymeXLAMLPassBase<LowerEnzymeXLAMLPass> {
  using Base::Base;

  void runOnOperation() override {
    lowerEnzymeXLAMath(getOperation(), [this]() { signalPassFailure(); });
  }
};
