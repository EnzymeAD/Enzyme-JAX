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
    auto loc = op.getLoc();
    auto operand = op.getOperand();

    auto zero = stablehlo::ConstantOp::create(
        rewriter, loc, cast<ElementsAttr>(makeAttr(op.getType(), 0)));
    auto maxTerm = stablehlo::MaxOp::create(rewriter, loc, operand, zero);

    rewriter.replaceOp(op, maxTerm);
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
    auto loc = op.getLoc();
    auto operand = op.getOperand();

    auto cstHalf = createConstantOpFromScalar(rewriter, op, 0.5);
    auto cstOne = createConstantOpFromScalar(rewriter, op, 1);
    auto cstInvSqrt2 =
        createConstantOpFromScalar(rewriter, op, 1 / std::sqrt(2));

    auto xOverSqrt2 =
        stablehlo::MulOp::create(rewriter, loc, operand, cstInvSqrt2);
    auto erfTerm = chlo::ErfOp::create(rewriter, loc, xOverSqrt2);
    auto onePlusErf = stablehlo::AddOp::create(rewriter, loc, cstOne, erfTerm);
    auto halfTimesOnePlusErf =
        stablehlo::MulOp::create(rewriter, loc, cstHalf, onePlusErf);
    auto result =
        stablehlo::MulOp::create(rewriter, loc, operand, halfTimesOnePlusErf);

    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult rewriteAsTanh(enzymexla::GeluOp op,
                              PatternRewriter &rewriter) const {
    // Ordering of the operations is important here. This comes from
    // https://github.com/openxla/xla/blob/3ab47f2c9324b10751ef18e58d2b303732685f30/xla/service/gpu/transforms/gemm_rewriter.cc#L773-L803
    // x * (0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    auto loc = op.getLoc();
    auto operand = op.getOperand();

    auto cstHalf = createConstantOpFromScalar(rewriter, op, 0.5);
    auto cstOne = createConstantOpFromScalar(rewriter, op, 1);
    auto cstSqrt2OverPi =
        createConstantOpFromScalar(rewriter, op, std::sqrt(2.0 / M_PI));
    auto cstCoeff = createConstantOpFromScalar(rewriter, op, 0.044715);

    auto x2 = stablehlo::MulOp::create(rewriter, loc, operand, operand);
    auto x3 = stablehlo::MulOp::create(rewriter, loc, operand, x2);
    auto coeffTimesX3 = stablehlo::MulOp::create(rewriter, loc, cstCoeff, x3);
    auto xPlusPoly =
        stablehlo::AddOp::create(rewriter, loc, operand, coeffTimesX3);
    auto scaled =
        stablehlo::MulOp::create(rewriter, loc, cstSqrt2OverPi, xPlusPoly);
    auto tanhTerm = stablehlo::TanhOp::create(rewriter, loc, scaled);
    auto onePlusTanh =
        stablehlo::AddOp::create(rewriter, loc, cstOne, tanhTerm);
    auto halfTimesOnePlusTanh =
        stablehlo::MulOp::create(rewriter, loc, cstHalf, onePlusTanh);
    auto result =
        stablehlo::MulOp::create(rewriter, loc, operand, halfTimesOnePlusTanh);

    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult rewriteAsSigmoid(enzymexla::GeluOp op,
                                 PatternRewriter &rewriter) const {
    // This is effectively the same as the tanh formulation but is typically
    // faster and more numerically accurate.
    // x * sigmoid(sqrt(8 / pi) * x * (1 + 0.044715 * x^2))
    auto loc = op.getLoc();
    auto operand = op.getOperand();

    auto cstSqrt8OverPi =
        createConstantOpFromScalar(rewriter, op, std::sqrt(8.0 / M_PI));
    auto cstOne = createConstantOpFromScalar(rewriter, op, 1);
    auto cstCoeff = createConstantOpFromScalar(rewriter, op, 0.044715);

    auto x2 = stablehlo::MulOp::create(rewriter, loc, operand, operand);
    auto coeffTimesX2 = stablehlo::MulOp::create(rewriter, loc, cstCoeff, x2);
    auto onePlusCoeffX2 =
        stablehlo::AddOp::create(rewriter, loc, cstOne, coeffTimesX2);
    auto xTimesInner =
        stablehlo::MulOp::create(rewriter, loc, operand, onePlusCoeffX2);
    auto scaled =
        stablehlo::MulOp::create(rewriter, loc, cstSqrt8OverPi, xTimesInner);
    auto sigmoid = stablehlo::LogisticOp::create(rewriter, loc, scaled);
    auto result = stablehlo::MulOp::create(rewriter, loc, operand, sigmoid);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerSoftplusOpToStablehlo
    : public OpRewritePattern<enzymexla::SoftplusOp> {
  using OpRewritePattern<enzymexla::SoftplusOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::SoftplusOp op,
                                PatternRewriter &rewriter) const override {
    // Numerically stable Softplus with NaN propagation:
    // softplus(x) = x                            if x != x (NaN)
    //               max(x, 0) + log1p(exp(-|x|)) otherwise
    auto loc = op.getLoc();
    auto operand = op.getOperand();

    auto zero = createConstantOpFromScalar(rewriter, op, 0);

    auto maxTerm = stablehlo::MaxOp::create(rewriter, loc, operand, zero);
    auto absOperand = stablehlo::AbsOp::create(rewriter, loc, operand);
    auto negAbsOperand = stablehlo::NegOp::create(rewriter, loc, absOperand);
    auto expNegAbs = stablehlo::ExpOp::create(rewriter, loc, negAbsOperand);
    auto logTerm = stablehlo::Log1pOp::create(rewriter, loc, expNegAbs);
    auto stableSoftplus =
        stablehlo::AddOp::create(rewriter, loc, maxTerm, logTerm);

    auto nanPred = stablehlo::CompareOp::create(
        rewriter, loc, operand, operand, stablehlo::ComparisonDirection::NE,
        stablehlo::ComparisonType::FLOAT);

    auto result = stablehlo::SelectOp::create(rewriter, loc, nanPred, operand,
                                              stableSoftplus);
    rewriter.replaceOp(op, result);
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
    patterns.add<LowerSoftplusOpToStablehlo>(context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // Verify that all illegal ops have been lowered
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa<enzymexla::ReluOp, enzymexla::GeluOp, enzymexla::SoftplusOp>(
              op)) {
        op->emitError("Failed to lower enzymexla ML operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
