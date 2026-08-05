//===- BatchNormExpanderPass.cpp - Expand StableHLO BatchNorm ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands StableHLO batch normalization operations
// (batch_norm_training, batch_norm_inference, batch_norm_grad) into primitive
// StableHLO operations. This is similar to XLA's batchnorm_expander.cc.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "batchnorm-expander"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_BATCHNORMEXPANDERPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;

namespace {

/// Create a scalar constant and broadcast it to the given shape.
static Value createBroadcastedScalar(PatternRewriter &rewriter, Location loc,
                                     RankedTensorType resultType,
                                     double scalarValue) {
  auto elementType = resultType.getElementType();
  auto scalarType = RankedTensorType::get({}, elementType);

  // Create scalar constant
  Attribute scalarAttr;
  if (isa<FloatType>(elementType)) {
    scalarAttr = rewriter.getFloatAttr(elementType, scalarValue);
  } else {
    scalarAttr = rewriter.getIntegerAttr(elementType, (int64_t)scalarValue);
  }
  auto scalarConst = ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(scalarType, scalarAttr));

  // Broadcast to result shape
  return BroadcastInDimOp::create(rewriter, loc, resultType, scalarConst,
                                  rewriter.getDenseI64ArrayAttr({}));
}

/// Get the dimensions to reduce over (all dimensions except feature_index)
static SmallVector<int64_t> getDimensionsWithoutFeature(int64_t rank,
                                                        int64_t featureIndex) {
  SmallVector<int64_t> dims;
  dims.reserve(rank - 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != featureIndex) {
      dims.push_back(i);
    }
  }
  return dims;
}

/// Compute the number of elements per feature (product of dimensions except
/// feature_index)
static int64_t computeElementsPerFeature(RankedTensorType operandType,
                                         int64_t featureIndex) {
  int64_t count = 1;
  for (int64_t i = 0; i < operandType.getRank(); ++i) {
    if (i != featureIndex) {
      count *= operandType.getDimSize(i);
    }
  }
  return count;
}

/// Expand stablehlo.batch_norm_training to primitive operations.
///
/// Output:
///   output = (operand - mean) / sqrt(variance + epsilon) * scale + offset
///   batch_mean = mean(operand) over non-feature dimensions
///   batch_var = var(operand) over non-feature dimensions
struct ExpandBatchNormTraining : public OpRewritePattern<BatchNormTrainingOp> {
  using OpRewritePattern<BatchNormTrainingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormTrainingOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    Value scale = op.getScale();
    Value offset = op.getOffset();
    float epsilon = op.getEpsilon().convertToFloat();
    int64_t featureIndex = op.getFeatureIndex();

    auto operandType = cast<RankedTensorType>(operand.getType());
    auto scaleType = cast<RankedTensorType>(scale.getType());
    Type elementType = operandType.getElementType();
    int64_t rank = operandType.getRank();

    auto dimsWithoutFeature = getDimensionsWithoutFeature(rank, featureIndex);
    int64_t elementsPerFeature =
        computeElementsPerFeature(operandType, featureIndex);

    // Create scalar type and zero attr for reduction init (fresh ones created
    // in lambda)
    auto scalarType = RankedTensorType::get({}, elementType);
    auto zeroAttr = rewriter.getFloatAttr(elementType, 0.0);

    // Create elements_per_feature constant
    auto elementsPerFeatureConst =
        createBroadcastedScalar(rewriter, loc, scaleType, elementsPerFeature);

    // Compute X^2
    auto operandSquared = MulOp::create(rewriter, loc, operand, operand);

    // Create reduce region (sum reduction)
    auto createSumReduce = [&](Value input) -> Value {
      // Create a fresh zero for each reduce (init value must be a separate SSA
      // value)
      auto freshZero = ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(scalarType, zeroAttr));

      auto reduceOp =
          ReduceOp::create(rewriter, loc, TypeRange{scaleType},
                           ValueRange{input}, ValueRange{freshZero},
                           rewriter.getDenseI64ArrayAttr(dimsWithoutFeature));

      // Build the reduction body - save insertion point first
      OpBuilder::InsertionGuard guard(rewriter);
      Block *reduceBody = rewriter.createBlock(&reduceOp.getBody());
      reduceBody->addArgument(scalarType, loc);
      reduceBody->addArgument(scalarType, loc);
      rewriter.setInsertionPointToStart(reduceBody);
      auto addResult = AddOp::create(rewriter, loc, reduceBody->getArgument(0),
                                     reduceBody->getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, ValueRange{addResult});

      return reduceOp.getResult(0);
    };

    // Sum[X]
    Value sum = createSumReduce(operand);

    // Sum[X^2]
    Value squaredSum = createSumReduce(operandSquared);

    // E[X] = Sum[X] / N
    auto mean = DivOp::create(rewriter, loc, sum, elementsPerFeatureConst);

    // E[X^2] = Sum[X^2] / N
    auto squareMean =
        DivOp::create(rewriter, loc, squaredSum, elementsPerFeatureConst);

    // E^2[X] = E[X] * E[X]
    auto meanSquare = MulOp::create(rewriter, loc, mean, mean);

    // Var[X] = E[X^2] - E^2[X]
    auto var = SubtractOp::create(rewriter, loc, squareMean, meanSquare);

    // Broadcast mean and var to operand shape
    SmallVector<int64_t> broadcastDims = {featureIndex};
    auto meanBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, operandType, mean,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));
    auto varBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, operandType, var,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));

    // epsilon broadcasted
    auto epsilonBroadcasted =
        createBroadcastedScalar(rewriter, loc, operandType, epsilon);

    // Var[X] + epsilon
    auto varAddEpsilon =
        AddOp::create(rewriter, loc, varBroadcasted, epsilonBroadcasted);

    // rsqrt(Var[X] + epsilon)
    auto rsqrtVarAddEpsilon = RsqrtOp::create(rewriter, loc, varAddEpsilon);

    // X - E[X]
    auto operandMinusMean =
        SubtractOp::create(rewriter, loc, operand, meanBroadcasted);

    // (X - E[X]) * rsqrt(Var[X] + epsilon)
    auto normalized =
        MulOp::create(rewriter, loc, operandMinusMean, rsqrtVarAddEpsilon);

    // Broadcast scale and offset to operand shape
    auto scaleBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, operandType, scale,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));
    auto offsetBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, operandType, offset,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));

    // normalized * scale
    auto scaledNormalized =
        MulOp::create(rewriter, loc, normalized, scaleBroadcasted);

    // normalized * scale + offset
    auto output =
        AddOp::create(rewriter, loc, scaledNormalized, offsetBroadcasted);

    rewriter.replaceOp(op, {output, mean, var});
    return success();
  }
};

/// Expand stablehlo.batch_norm_inference to primitive operations.
///
/// Output:
///   output = (operand - mean) / sqrt(variance + epsilon) * scale + offset
///
/// Where mean and variance are provided as inputs (not computed).
struct ExpandBatchNormInference
    : public OpRewritePattern<BatchNormInferenceOp> {
  using OpRewritePattern<BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormInferenceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    Value scale = op.getScale();
    Value offset = op.getOffset();
    Value mean = op.getMean();
    Value variance = op.getVariance();
    float epsilon = op.getEpsilon().convertToFloat();
    int64_t featureIndex = op.getFeatureIndex();

    auto operandType = cast<RankedTensorType>(operand.getType());
    auto scaleType = cast<RankedTensorType>(scale.getType());

    SmallVector<int64_t> broadcastDims = {featureIndex};

    // epsilon constant (broadcasted to scale shape first)
    auto epsilonConst =
        createBroadcastedScalar(rewriter, loc, scaleType, epsilon);

    // variance + epsilon
    auto varAddEpsilon = AddOp::create(rewriter, loc, variance, epsilonConst);

    // rsqrt(variance + epsilon)
    auto rsqrtVarAddEpsilon = RsqrtOp::create(rewriter, loc, varAddEpsilon);

    // true_scale = scale * rsqrt(variance + epsilon)
    auto trueScale = MulOp::create(rewriter, loc, scale, rsqrtVarAddEpsilon);

    // true_shift = offset - mean * true_scale
    auto meanTimesScale = MulOp::create(rewriter, loc, mean, trueScale);
    auto trueShift = SubtractOp::create(rewriter, loc, offset, meanTimesScale);

    // Broadcast true_scale and true_shift to operand shape
    auto trueScaleBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, operandType, trueScale,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));
    auto trueShiftBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, operandType, trueShift,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));

    // output = operand * true_scale + true_shift
    auto scaled = MulOp::create(rewriter, loc, operand, trueScaleBroadcasted);
    auto output = AddOp::create(rewriter, loc, scaled, trueShiftBroadcasted);

    rewriter.replaceOp(op, ValueRange{output});
    return success();
  }
};

/// Expand stablehlo.batch_norm_grad to primitive operations.
///
/// Gradients:
///   scale_grad = sum(output_grad * (activation - mean(activation))) *
///                rsqrt(var + epsilon)
///   offset_grad = sum(output_grad)
///   activation_grad = 1/N * scale * rsqrt(var + epsilon) *
///                     (N * output_grad - sum(output_grad) -
///                      (activation - mean(activation)) *
///                      sum(output_grad * (activation - mean(activation))) /
///                      (variance + epsilon))
struct ExpandBatchNormGrad : public OpRewritePattern<BatchNormGradOp> {
  using OpRewritePattern<BatchNormGradOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormGradOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value activation = op.getOperand();
    Value scale = op.getScale();
    Value mean = op.getMean();
    Value variance = op.getVariance();
    Value gradOutput = op.getGradOutput();
    float epsilon = op.getEpsilon().convertToFloat();
    int64_t featureIndex = op.getFeatureIndex();

    auto activationType = cast<RankedTensorType>(activation.getType());
    auto scaleType = cast<RankedTensorType>(scale.getType());
    Type elementType = activationType.getElementType();
    int64_t rank = activationType.getRank();

    auto dimsWithoutFeature = getDimensionsWithoutFeature(rank, featureIndex);
    int64_t elementsPerFeature =
        computeElementsPerFeature(activationType, featureIndex);

    SmallVector<int64_t> broadcastDims = {featureIndex};

    // Create scalar type and zero attr for reduction init (fresh ones created
    // in lambda)
    auto scalarType = RankedTensorType::get({}, elementType);
    auto zeroAttr = rewriter.getFloatAttr(elementType, 0.0);

    // Create sum reduction helper
    auto createSumReduce = [&](Value input) -> Value {
      // Create a fresh zero for each reduce (init value must be a separate SSA
      // value)
      auto freshZero = ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(scalarType, zeroAttr));

      auto reduceOp =
          ReduceOp::create(rewriter, loc, TypeRange{scaleType},
                           ValueRange{input}, ValueRange{freshZero},
                           rewriter.getDenseI64ArrayAttr(dimsWithoutFeature));

      // Build the reduction body - save insertion point first
      OpBuilder::InsertionGuard guard(rewriter);
      Block *reduceBody = rewriter.createBlock(&reduceOp.getBody());
      reduceBody->addArgument(scalarType, loc);
      reduceBody->addArgument(scalarType, loc);
      rewriter.setInsertionPointToStart(reduceBody);
      auto addResult = AddOp::create(rewriter, loc, reduceBody->getArgument(0),
                                     reduceBody->getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, ValueRange{addResult});

      return reduceOp.getResult(0);
    };

    // Broadcast scale, mean, variance to activation shape
    auto scaleBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, activationType, scale,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));
    auto meanBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, activationType, mean,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));
    auto varianceBroadcasted =
        BroadcastInDimOp::create(rewriter, loc, activationType, variance,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));

    // epsilon constants
    auto epsilonActivation =
        createBroadcastedScalar(rewriter, loc, activationType, epsilon);
    auto epsilonFeature =
        createBroadcastedScalar(rewriter, loc, scaleType, epsilon);

    // rsqrt(variance + epsilon) for both shapes
    auto varAddEpsilonBroadcasted =
        AddOp::create(rewriter, loc, varianceBroadcasted, epsilonActivation);
    auto rsqrtVarAddEpsilonBroadcasted =
        RsqrtOp::create(rewriter, loc, varAddEpsilonBroadcasted);

    auto varAddEpsilonFeature =
        AddOp::create(rewriter, loc, variance, epsilonFeature);
    auto rsqrtVarAddEpsilonFeature =
        RsqrtOp::create(rewriter, loc, varAddEpsilonFeature);

    // X - E[X]
    auto activationMinusMean =
        SubtractOp::create(rewriter, loc, activation, meanBroadcasted);

    // Grad[Y] * (X - E[X])
    auto gradOutputTimesActivationMinusMean =
        MulOp::create(rewriter, loc, gradOutput, activationMinusMean);

    // sum(Grad[Y] * (X - E[X]))
    auto sumGradOutputTimesActivationMinusMean =
        createSumReduce(gradOutputTimesActivationMinusMean);

    // Grad[beta] = Sum(Grad[Y])
    auto gradBeta = createSumReduce(gradOutput);

    // Grad[scale] = Sum(Grad[Y] * (X - E[X]) * rsqrt[Var[X] + epsilon])
    auto gradScale =
        MulOp::create(rewriter, loc, sumGradOutputTimesActivationMinusMean,
                      rsqrtVarAddEpsilonFeature);

    // I2 = broadcast(Sum(Grad[Y]))
    auto i2 =
        BroadcastInDimOp::create(rewriter, loc, activationType, gradBeta,
                                 rewriter.getDenseI64ArrayAttr(broadcastDims));

    // I3 = broadcast(Sum(Grad[Y] * (X - E[X])))
    auto i3 = BroadcastInDimOp::create(
        rewriter, loc, activationType, sumGradOutputTimesActivationMinusMean,
        rewriter.getDenseI64ArrayAttr(broadcastDims));

    // I4 = (X - E[X]) * I3
    auto i4 = MulOp::create(rewriter, loc, activationMinusMean, i3);

    // I5 = I4 / (Var[X] + epsilon)
    auto i5 = DivOp::create(rewriter, loc, i4, varAddEpsilonBroadcasted);

    // N constant broadcasted
    auto nBroadcasted = createBroadcastedScalar(rewriter, loc, activationType,
                                                elementsPerFeature);

    // scale * rsqrt[Var[X] + epsilon] / N
    auto scaleTimesRsqrt = MulOp::create(rewriter, loc, scaleBroadcasted,
                                         rsqrtVarAddEpsilonBroadcasted);
    auto scaleTimesRsqrtDivN =
        DivOp::create(rewriter, loc, scaleTimesRsqrt, nBroadcasted);

    // I1 = N * Grad[Y]
    auto i1 = MulOp::create(rewriter, loc, gradOutput, nBroadcasted);

    // I6 = I1 - I2 - I5
    auto i1MinusI2 = SubtractOp::create(rewriter, loc, i1, i2);
    auto i6 = SubtractOp::create(rewriter, loc, i1MinusI2, i5);

    // Grad[X] = scale * rsqrt[Var[X] + epsilon] * 1/N * I6
    auto gradActivation = MulOp::create(rewriter, loc, scaleTimesRsqrtDivN, i6);

    rewriter.replaceOp(op, {gradActivation, gradScale, gradBeta});
    return success();
  }
};

struct BatchNormExpanderPass
    : public enzyme::impl::BatchNormExpanderPassBase<BatchNormExpanderPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add patterns based on options
    if (rewriteTrainingOp) {
      patterns.add<ExpandBatchNormTraining>(context);
    }
    if (rewriteInferenceOp) {
      patterns.add<ExpandBatchNormInference>(context);
    }
    if (rewriteGradOp) {
      patterns.add<ExpandBatchNormGrad>(context);
    }

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};

} // namespace
