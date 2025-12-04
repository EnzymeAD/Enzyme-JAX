// This must come first for windows builds
#define _USE_MATH_DEFINES

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla-ml"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

struct LowerReluOpToStablehlo : public OpRewritePattern<enzymexla::ReluOp> {
  using OpRewritePattern<enzymexla::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::ReluOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(
        op, op.getOperand(),
        stablehlo::ConstantOp::create(
            rewriter, op.getLoc(),
            cast<ElementsAttr>(makeAttr(op.getType(), 0))));
    return success();
  }
};

struct LowerGeluOpToStablehlo : public OpRewritePattern<enzymexla::GeluOp> {
  using OpRewritePattern<enzymexla::GeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::GeluOp op,
                                PatternRewriter &rewriter) const override {
    switch (op.getGeluApproximation()) {
    case enzymexla::GeluApproximation::NONE:
      return rewriteAsErf(op, rewriter);
    case enzymexla::GeluApproximation::TANH:
      return rewriteAsTanh(op, rewriter);
    case enzymexla::GeluApproximation::SIGMOID:
      return rewriteAsSigmoid(op, rewriter);
    }

    return failure();
  }

private:
  LogicalResult rewriteAsErf(enzymexla::GeluOp op,
                             PatternRewriter &rewriter) const {
    // x * (0.5 * (1 + erf(x / sqrt(2))))
    auto operand = op.getOperand();
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
        op, operand,
        stablehlo::MulOp::create(
            rewriter, op.getLoc(),
            createConstantOpFromScalar(rewriter, op, 0.5),
            stablehlo::AddOp::create(
                rewriter, op.getLoc(),
                createConstantOpFromScalar(rewriter, op, 1),
                chlo::ErfOp::create(
                    rewriter, op.getLoc(),
                    stablehlo::MulOp::create(
                        rewriter, op.getLoc(), operand,
                        createConstantOpFromScalar(rewriter, op,
                                                   1 / std::sqrt(2)))))));
    return success();
  }

  LogicalResult rewriteAsTanh(enzymexla::GeluOp op,
                              PatternRewriter &rewriter) const {
    // Ordering of the operations is important here. This comes from
    // https://github.com/openxla/xla/blob/3ab47f2c9324b10751ef18e58d2b303732685f30/xla/service/gpu/transforms/gemm_rewriter.cc#L773-L803
    // x * (0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    auto operand = op.getOperand();
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
        op, operand,
        stablehlo::MulOp::create(
            rewriter, op.getLoc(),
            createConstantOpFromScalar(rewriter, op, 0.5),
            stablehlo::AddOp::create(
                rewriter, op.getLoc(),
                createConstantOpFromScalar(rewriter, op, 1),
                stablehlo::TanhOp::create(
                    rewriter, op.getLoc(),
                    stablehlo::MulOp::create(
                        rewriter, op.getLoc(),
                        createConstantOpFromScalar(rewriter, op,
                                                   std::sqrt(2.0 / M_PI)),
                        stablehlo::AddOp::create(
                            rewriter, op.getLoc(), operand,
                            stablehlo::MulOp::create(
                                rewriter, op.getLoc(),
                                createConstantOpFromScalar(rewriter, op,
                                                           0.044715),
                                stablehlo::MulOp::create(
                                    rewriter, op.getLoc(), operand,
                                    stablehlo::MulOp::create(
                                        rewriter, op.getLoc(), operand,
                                        operand)))))))));
    return success();
  }

  LogicalResult rewriteAsSigmoid(enzymexla::GeluOp op,
                                 PatternRewriter &rewriter) const {
    // This is effectively the same as the tanh formulation but is typically
    // faster and more numerically accurate.
    // x * sigmoid(sqrt(8 / pi) * x * (1 + 0.044715 * x^2))
    auto operand = op.getOperand();
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
        op, operand,
        stablehlo::LogisticOp::create(
            rewriter, op.getLoc(),
            stablehlo::MulOp::create(
                rewriter, op.getLoc(),
                createConstantOpFromScalar(rewriter, op, std::sqrt(8.0 / M_PI)),
                stablehlo::MulOp::create(
                    rewriter, op.getLoc(), operand,
                    stablehlo::AddOp::create(
                        rewriter, op.getLoc(),
                        createConstantOpFromScalar(rewriter, op, 1),
                        stablehlo::MulOp::create(
                            rewriter, op.getLoc(),
                            createConstantOpFromScalar(rewriter, op, 0.044715),
                            stablehlo::MulOp::create(rewriter, op.getLoc(),
                                                     operand, operand)))))));
    return success();
  }
};

struct LowerEnzymeXLAMLPass
    : public enzyme::impl::LowerEnzymeXLAMLPassBase<LowerEnzymeXLAMLPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    // SHLO Lowering
    patterns.add<LowerReluOpToStablehlo>(context);
    patterns.add<LowerGeluOpToStablehlo>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
