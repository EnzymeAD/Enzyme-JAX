#include "Enzyme/MLIR/Analysis/SampleDependenceAnalysis.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "sicm"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SICMPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

static bool isScalar(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return false;
  return tensorType.getNumElements() == 1;
}

static std::pair<Value, Value>
matchScaledInvariant(Value value, SampleDependenceAnalysis &analysis) {
  auto mulOp = value.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp)
    return {nullptr, nullptr};

  Value lhs = mulOp.getLhs();
  Value rhs = mulOp.getRhs();

  auto lhsBroadcast = lhs.getDefiningOp<stablehlo::BroadcastInDimOp>();
  auto rhsBroadcast = rhs.getDefiningOp<stablehlo::BroadcastInDimOp>();

  Value scale, invariant;
  if (lhsBroadcast && isScalar(lhsBroadcast.getOperand())) {
    scale = lhsBroadcast.getOperand();
    invariant = rhs;
  } else if (rhsBroadcast && isScalar(rhsBroadcast.getOperand())) {
    scale = rhsBroadcast.getOperand();
    invariant = lhs;
  } else if (isScalar(lhs) && !isScalar(rhs)) {
    scale = lhs;
    invariant = rhs;
  } else if (isScalar(rhs) && !isScalar(lhs)) {
    scale = rhs;
    invariant = lhs;
  } else {
    return {nullptr, nullptr};
  }

  if (analysis.isSampleDependent(invariant))
    return {nullptr, nullptr};

  return {scale, invariant};
}

static Value broadcastScalarToShape(OpBuilder &rewriter, Location loc,
                                    Value scalar, RankedTensorType resultType) {
  auto scaleType = cast<RankedTensorType>(scalar.getType());
  Value scalarVal = scalar;
  if (scaleType.getRank() > 0) {
    auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
    scalarVal = stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scalar);
  }

  auto scalarValType = cast<RankedTensorType>(scalarVal.getType());
  if (scalarValType.getShape() != resultType.getShape()) {
    SmallVector<int64_t> broadcastDims;
    return stablehlo::BroadcastInDimOp::create(
        rewriter, loc, resultType, scalarVal,
        rewriter.getDenseI64ArrayAttr(broadcastDims));
  }
  return scalarVal;
}

/// TODO: Need to prove positive scale
struct CholeskyScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::CholeskyOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  CholeskyScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                                bool &patternApplied, MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp choleskyOp,
                                PatternRewriter &rewriter) const override {
    // Only match within the analyzed mcmc_region
    if (!analysis.isInTargetRegion(choleskyOp))
      return failure();

    Location loc = choleskyOp.getLoc();
    Value input = choleskyOp.getA();

    auto [scale, baseCov] = matchScaledInvariant(input, analysis);
    if (!scale)
      return failure();

    // Create sqrt(scale) as a rank-0 scalar
    auto scaleType = cast<RankedTensorType>(scale.getType());
    Value scalarScale = scale;
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto sqrtScale =
        math::SqrtOp::create(rewriter, loc, scalarScale.getType(), scalarScale);

    // Create cholesky(baseCov) — now sample-invariant, will be hoisted
    auto newCholesky =
        stablehlo::CholeskyOp::create(rewriter, loc, choleskyOp.getType(),
                                      baseCov, choleskyOp.getLowerAttr());

    // Broadcast sqrt(scale) to result shape and multiply
    auto resultType = cast<RankedTensorType>(choleskyOp.getType());
    Value broadcastedSqrt =
        broadcastScalarToShape(rewriter, loc, sqrtScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedSqrt, newCholesky);

    rewriter.replaceOp(choleskyOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

struct OuterProductInfo {
  Value sourceVec;
  Value rowBroadcast;
  Value colBroadcast;
};

static Value lookThroughReshapes(Value v) {
  while (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>())
    v = reshape.getOperand();
  return v;
}

static OuterProductInfo matchOuterProduct(Value value) {
  auto mulOp = value.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp)
    return {nullptr, nullptr, nullptr};

  auto lBcast = mulOp.getLhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
  auto rBcast = mulOp.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
  if (!lBcast || !rBcast)
    return {nullptr, nullptr, nullptr};

  Value lSource = lookThroughReshapes(lBcast.getOperand());
  Value rSource = lookThroughReshapes(rBcast.getOperand());
  if (lSource != rSource)
    return {nullptr, nullptr, nullptr};

  if (lBcast.getOperand() == rBcast.getOperand() &&
      lBcast.getBroadcastDimensions() == rBcast.getBroadcastDimensions())
    return {nullptr, nullptr, nullptr};

  Value sourceVec = lSource;

  Value rowBcast = nullptr, colBcast = nullptr;
  for (auto bcast : {lBcast, rBcast}) {
    auto bcastSourceType = cast<RankedTensorType>(bcast.getOperand().getType());
    auto dims = bcast.getBroadcastDimensions();
    bool isRow = false;
    for (int64_t i = 0; i < bcastSourceType.getRank(); ++i) {
      if (bcastSourceType.getDimSize(i) > 1) {
        isRow = (dims[i] == 0);
        break;
      }
    }
    if (isRow)
      rowBcast = bcast.getResult();
    else
      colBcast = bcast.getResult();
  }

  if (!rowBcast || !colBcast)
    return {nullptr, nullptr, nullptr};

  return {sourceVec, rowBcast, colBcast};
}

/// Cholesky factorization for diagonal-scaled invariant correlation matrix.
///
/// Rewrites cholesky(D ⊙ Ω) where:
///   D = outer_product(σ, σ) = σ_i * σ_j (diagonal scale matrix)
///   Ω = sample-invariant correlation matrix
///
/// Identity: chol(diag(σ) Ω diag(σ)) = diag(σ) chol(Ω)  [lower=true]
///           chol(diag(σ) Ω diag(σ)) = chol(Ω) diag(σ)  [lower=false]
///
/// The chol(Ω) is sample-invariant and will be auto-hoisted in Phase 2.
struct CholeskyOuterProductScaleHLO
    : public OpRewritePattern<stablehlo::CholeskyOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  CholeskyOuterProductScaleHLO(SampleDependenceAnalysis &analysis,
                               bool &patternApplied, MLIRContext *context,
                               PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp cholOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(cholOp))
      return failure();

    // Match: multiply(outer_product(σ, σ), Ω) where Ω is invariant
    auto mulOp = cholOp.getA().getDefiningOp<stablehlo::MulOp>();
    if (!mulOp)
      return failure();

    OuterProductInfo outerProd;
    Value invariantMatrix;

    for (auto [maybeDiag, maybeInvariant] :
         {std::pair{mulOp.getLhs(), mulOp.getRhs()},
          std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
      if (analysis.isSampleDependent(maybeInvariant))
        continue;
      auto op = matchOuterProduct(maybeDiag);
      if (!op.sourceVec)
        continue;
      outerProd = op;
      invariantMatrix = maybeInvariant;
      break;
    }

    if (!outerProd.sourceVec)
      return failure();

    Location loc = cholOp.getLoc();
    auto inputType = cast<RankedTensorType>(cholOp.getType());

    // chol(Ω) — invariant, will be hoisted
    auto newChol = stablehlo::CholeskyOp::create(
        rewriter, loc, inputType, invariantMatrix, cholOp.getLowerAttr());

    Value scaleBcast =
        cholOp.getLower() ? outerProd.rowBroadcast : outerProd.colBroadcast;
    Value result =
        stablehlo::MulOp::create(rewriter, loc, inputType, scaleBcast, newChol);

    rewriter.replaceOp(cholOp, result);
    patternApplied = true;
    return success();
  }
};

struct DotGeneralScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  DotGeneralScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                                  bool &patternApplied, MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(dotOp))
      return failure();

    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();

    // Try to match scaled invariant on LHS first, then RHS
    Value scale, invariant, other;
    bool scaledOnLhs = false;

    auto [lhsScale, lhsInvariant] = matchScaledInvariant(lhs, analysis);
    if (lhsScale) {
      scale = lhsScale;
      invariant = lhsInvariant;
      other = rhs;
      scaledOnLhs = true;
    } else {
      auto [rhsScale, rhsInvariant] = matchScaledInvariant(rhs, analysis);
      if (!rhsScale)
        return failure();
      scale = rhsScale;
      invariant = rhsInvariant;
      other = lhs;
      scaledOnLhs = false;
    }

    // Build the new dot_general with the invariant operand replacing the scaled
    auto resultType = cast<RankedTensorType>(dotOp.getType());
    Value newLhs = scaledOnLhs ? invariant : other;
    Value newRhs = scaledOnLhs ? other : invariant;

    auto newDot = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, newRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());

    // Broadcast scalar to result shape and multiply
    Value broadcastedScale =
        broadcastScalarToShape(rewriter, loc, scale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedScale, newDot);

    rewriter.replaceOp(dotOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// Triangular solve scale factorization pattern for StableHLO.
/// triangular_solve(broadcast(s) * L, b) -> (1/s) * triangular_solve(L, b)
/// when both L and b are sample-invariant.
struct TriangularSolveScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::TriangularSolveOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  TriangularSolveScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                                       bool &patternApplied,
                                       MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::TriangularSolveOp solveOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(solveOp))
      return failure();

    Location loc = solveOp.getLoc();
    Value a = solveOp.getA();
    Value b = solveOp.getB();

    // Match: a = broadcast(scalar) * invariant_L
    auto [scale, baseL] = matchScaledInvariant(a, analysis);
    if (!scale)
      return failure();

    // b must also be sample-invariant for the solve to be hoistable
    if (analysis.isSampleDependent(b))
      return failure();

    // Create triangular_solve(L_base, b) — now fully sample-invariant
    auto resultType = cast<RankedTensorType>(solveOp.getType());
    auto newSolve = stablehlo::TriangularSolveOp::create(
        rewriter, loc, resultType, baseL, b, solveOp.getLeftSideAttr(),
        solveOp.getLowerAttr(), solveOp.getUnitDiagonalAttr(),
        solveOp.getTransposeAAttr());

    // Compute 1/s
    auto scaleType = cast<RankedTensorType>(scale.getType());
    auto elemType = scaleType.getElementType();
    auto scalarType = RankedTensorType::get({}, elemType);
    Value one = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(scalarType,
                               rewriter.getFloatAttr(elemType, 1.0)));
    Value scalarScale = scale;
    if (scaleType.getRank() > 0)
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    Value invScale =
        stablehlo::DivOp::create(rewriter, loc, scalarType, one, scalarScale);

    // Broadcast (1/s) to result shape and multiply
    Value broadcastedInvScale =
        broadcastScalarToShape(rewriter, loc, invScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedInvScale, newSolve);

    rewriter.replaceOp(solveOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// Log-multiply distribution pattern for StableHLO.
/// log(broadcast(s) * A) -> broadcast(log(s)) + log(A)
/// when A is sample-invariant.
struct LogMultiplyDistributionHLO : public OpRewritePattern<stablehlo::LogOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  LogMultiplyDistributionHLO(SampleDependenceAnalysis &analysis,
                             bool &patternApplied, MLIRContext *context,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::LogOp logOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(logOp))
      return failure();

    Location loc = logOp.getLoc();
    Value input = logOp.getOperand();

    // Match: input = broadcast(scalar) * invariant
    auto [scale, invariant] = matchScaledInvariant(input, analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(logOp.getType());

    // log(invariant) — sample-invariant, will be hoisted
    auto logInvariant =
        stablehlo::LogOp::create(rewriter, loc, resultType, invariant);

    // log(scalar)
    auto scaleType = cast<RankedTensorType>(scale.getType());
    Value scalarScale = scale;
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto logScale = stablehlo::LogOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);

    // Broadcast log(scalar) to result shape and add
    Value broadcastedLogScale =
        broadcastScalarToShape(rewriter, loc, logScale, resultType);
    auto result = stablehlo::AddOp::create(rewriter, loc, resultType,
                                           broadcastedLogScale, logInvariant);

    rewriter.replaceOp(logOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DotAbsorb patterns: absorb invariant ops from RHS into LHS matrix
//===----------------------------------------------------------------------===//

static std::pair<Value, stablehlo::ReshapeOp> lookThroughReshape(Value rhs) {
  if (auto reshapeOp = rhs.getDefiningOp<stablehlo::ReshapeOp>())
    return {reshapeOp.getOperand(), reshapeOp};
  return {rhs, nullptr};
}

/// DotAbsorbDiagMulHLO: Absorb a per-column diagonal multiply into the LHS.
/// dot_general(A, multiply(X, broadcast(c)))
///   -> dot_general(multiply(A, broadcast(c)), X)
/// where A and c are sample-invariant.
struct DotAbsorbDiagMulHLO : public OpRewritePattern<stablehlo::DotGeneralOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  DotAbsorbDiagMulHLO(SampleDependenceAnalysis &analysis, bool &patternApplied,
                      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(dotOp))
      return failure();

    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();

    if (analysis.isSampleDependent(lhs))
      return failure();

    // Match: rhs = multiply(X, broadcast(c)) or multiply(broadcast(c), X)
    auto mulOp = rhs.getDefiningOp<stablehlo::MulOp>();
    if (!mulOp)
      return failure();

    Value mulLhs = mulOp.getLhs();
    Value mulRhs = mulOp.getRhs();

    Value scale, inner;
    stablehlo::BroadcastInDimOp broadcast;
    if (auto bcast = mulLhs.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
      if (!analysis.isSampleDependent(bcast.getOperand())) {
        broadcast = bcast;
        scale = bcast.getOperand();
        inner = mulRhs;
      }
    }
    if (!scale) {
      if (auto bcast = mulRhs.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
        if (!analysis.isSampleDependent(bcast.getOperand())) {
          broadcast = bcast;
          scale = bcast.getOperand();
          inner = mulLhs;
        }
      }
    }
    if (!scale)
      return failure();

    // Broadcast scale to LHS shape along the contracting dimension
    auto lhsType = cast<RankedTensorType>(lhs.getType());

    // Get contracting dims from the dot
    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();

    // The broadcast on the RHS broadcasts scale along some dims.
    // We need to broadcast the same scale to LHS shape along the LHS
    // contracting dims (which correspond to the RHS contracting dims).
    auto rhsBcastDims = broadcast.getBroadcastDimensions();

    // Build broadcast dims for LHS: map RHS dims to LHS dims.
    // The contracting dims pair up: lhsContractDims[i] <-> rhsContractDims[i].
    auto rhsContractDims = dimNumbers.getRhsContractingDimensions();

    // Build a mapping from RHS dim -> LHS dim
    DenseMap<int64_t, int64_t> rhsToLhs;
    for (size_t i = 0; i < rhsContractDims.size(); ++i)
      rhsToLhs[rhsContractDims[i]] = lhsContractDims[i];

    // Map the broadcast dims from RHS space to LHS space
    SmallVector<int64_t> lhsBcastDims;
    for (int64_t d : rhsBcastDims) {
      auto it = rhsToLhs.find(d);
      if (it == rhsToLhs.end())
        return failure(); // broadcast dim is not a contracting dim
      lhsBcastDims.push_back(it->second);
    }

    // Broadcast scale to LHS shape
    Value lhsBroadcasted = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, lhsType, scale,
        rewriter.getDenseI64ArrayAttr(lhsBcastDims));

    // Multiply: A * broadcast(c)
    Value newLhs =
        stablehlo::MulOp::create(rewriter, loc, lhsType, lhs, lhsBroadcasted);

    // Replace dot with dot_general(A * broadcast(c), X)
    auto resultType = cast<RankedTensorType>(dotOp.getType());
    auto newDot = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, inner,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());

    rewriter.replaceOp(dotOp, newDot.getResult());
    patternApplied = true;
    return success();
  }
};

/// DotAbsorbFFTHLO: Absorb an FFT from the RHS into the LHS matrix.
/// dot_general(A:[M,P], reshape(fft(X)))  ->  dot_general(A':[M,P], reshape(X))
/// where A' = reshape(fft(reshape(A,[M,N1,...,Nk])), [M,P])
/// and A is sample-invariant.
struct DotAbsorbFFTHLO : public OpRewritePattern<stablehlo::DotGeneralOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  DotAbsorbFFTHLO(SampleDependenceAnalysis &analysis, bool &patternApplied,
                  MLIRContext *context, PatternBenefit benefit = 4)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(dotOp))
      return failure();

    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();

    if (analysis.isSampleDependent(lhs))
      return failure();

    // Look through reshape on RHS
    auto [rhsInner, reshapeOp] = lookThroughReshape(rhs);

    // Match FFT
    auto fftOp = rhsInner.getDefiningOp<stablehlo::FftOp>();
    if (!fftOp)
      return failure();

    Value fftInput = fftOp.getOperand();
    auto fftLength = fftOp.getFftLength();
    auto fftType = fftOp.getFftType();

    // Get LHS type info
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsShape = lhsType.getShape();
    int64_t lhsRank = lhsType.getRank();

    // Reshape A from [M, P] to [M, N1, ..., Nk] where P = N1*...*Nk
    // and [N1,...,Nk] are the FFT dimensions
    SmallVector<int64_t> fftDims(fftLength.begin(), fftLength.end());

    // Build the expanded LHS shape: batch dims + fft dims
    // The contracting dims of LHS correspond to the product of fft dims
    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();

    // Handle single contracting dim on LHS (first or last dim)
    if (lhsContractDims.size() != 1)
      return failure();
    int64_t contractDim = lhsContractDims[0];
    if (contractDim != 0 && contractDim != lhsRank - 1)
      return failure();

    int64_t P = lhsShape[contractDim];

    // Verify P matches the product of FFT dims
    int64_t fftProduct = 1;
    for (int64_t d : fftDims)
      fftProduct *= d;
    if (P != fftProduct)
      return failure();

    // Build expanded LHS shape: [batchBefore..., N1,...,Nk, batchAfter...]
    SmallVector<int64_t> expandedLhsShape;
    for (int64_t i = 0; i < contractDim; ++i)
      expandedLhsShape.push_back(lhsShape[i]);
    for (int64_t d : fftDims)
      expandedLhsShape.push_back(d);
    for (int64_t i = contractDim + 1; i < lhsRank; ++i)
      expandedLhsShape.push_back(lhsShape[i]);

    auto expandedLhsType =
        RankedTensorType::get(expandedLhsShape, lhsType.getElementType());
    Value expandedLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, expandedLhsType, lhs);

    // StableHLO FFT operates on trailing len(fft_length) dims.
    // When contractDim == 0, the FFT dims are at the front after reshape,
    // so we need transposes to move them to trailing position.
    int64_t numFftDims = fftDims.size();
    int64_t numBatchAfter = lhsRank - contractDim - 1;
    Value toFFT = expandedLhs;

    if (contractDim == 0 && numBatchAfter > 0) {
      // Transpose: [N1,...,Nk, batch...] -> [batch..., N1,...,Nk]
      SmallVector<int64_t> perm;
      for (int64_t i = numFftDims; i < (int64_t)expandedLhsShape.size(); ++i)
        perm.push_back(i);
      for (int64_t i = 0; i < numFftDims; ++i)
        perm.push_back(i);

      SmallVector<int64_t> transposedShape;
      for (int64_t p : perm)
        transposedShape.push_back(expandedLhsShape[p]);

      auto transposedType =
          RankedTensorType::get(transposedShape, lhsType.getElementType());
      toFFT = stablehlo::TransposeOp::create(
          rewriter, loc, transposedType, expandedLhs,
          rewriter.getDenseI64ArrayAttr(perm));
    }

    // Apply FFT on trailing dims (works for both dim-0 and last-dim cases)
    auto fftedLhs = stablehlo::FftOp::create(
        rewriter, loc, toFFT, fftType, rewriter.getDenseI64ArrayAttr(fftDims));

    Value reshapeInput = fftedLhs;
    if (contractDim == 0 && numBatchAfter > 0) {
      // Transpose back: [batch..., N1,...,Nk] -> [N1,...,Nk, batch...]
      SmallVector<int64_t> invPerm;
      for (int64_t i = numBatchAfter; i < (int64_t)expandedLhsShape.size(); ++i)
        invPerm.push_back(i);
      for (int64_t i = 0; i < numBatchAfter; ++i)
        invPerm.push_back(i);

      auto expandedType =
          RankedTensorType::get(expandedLhsShape, lhsType.getElementType());
      reshapeInput = stablehlo::TransposeOp::create(
          rewriter, loc, expandedType, fftedLhs,
          rewriter.getDenseI64ArrayAttr(invPerm));
    }

    // Reshape back to original LHS shape
    Value newLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, lhsType, reshapeInput);

    // Replace the RHS: remove the FFT, keep the reshape
    Value newRhs;
    if (reshapeOp) {
      auto reshapeType = cast<RankedTensorType>(reshapeOp.getType());
      newRhs =
          stablehlo::ReshapeOp::create(rewriter, loc, reshapeType, fftInput);
    } else {
      newRhs = fftInput;
    }

    auto resultType = cast<RankedTensorType>(dotOp.getType());
    auto newDot = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, newRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());

    rewriter.replaceOp(dotOp, newDot.getResult());
    patternApplied = true;
    return success();
  }
};

/// DotAbsorbTransposeHLO: Absorb a transpose from the RHS into the LHS matrix.
/// dot_general(A:[M,P], reshape(transpose(X, perm)))
///   -> dot_general(A':[M,P], reshape(X))
/// where A' = reshape(transpose(reshape(A, [M, perm_shape...]), [0,
/// inv_perm...]), [M,P]) and A is sample-invariant.
struct DotAbsorbTransposeHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  DotAbsorbTransposeHLO(SampleDependenceAnalysis &analysis,
                        bool &patternApplied, MLIRContext *context,
                        PatternBenefit benefit = 3)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(dotOp))
      return failure();

    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();

    if (analysis.isSampleDependent(lhs))
      return failure();

    // Look through reshape on RHS
    auto [rhsInner, reshapeOp] = lookThroughReshape(rhs);

    // Match transpose
    auto transposeOp = rhsInner.getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();

    Value transposeInput = transposeOp.getOperand();
    auto transposeInputType = cast<RankedTensorType>(transposeInput.getType());
    auto inputShape = transposeInputType.getShape();
    auto perm = transposeOp.getPermutation();

    // Get LHS type info
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsShape = lhsType.getShape();
    int64_t lhsRank = lhsType.getRank();

    // Handle single contracting dim on LHS (first or last dim)
    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();
    if (lhsContractDims.size() != 1)
      return failure();
    int64_t contractDim = lhsContractDims[0];
    if (contractDim != 0 && contractDim != lhsRank - 1)
      return failure();

    int64_t P = lhsShape[contractDim];

    // Verify P matches the product of input dims
    int64_t inputProduct = 1;
    for (int64_t d : inputShape)
      inputProduct *= d;
    if (P != inputProduct)
      return failure();

    // Build expanded LHS shape: [batchBefore..., transposedShape...,
    // batchAfter...]
    SmallVector<int64_t> expandedLhsShape;
    for (int64_t i = 0; i < contractDim; ++i)
      expandedLhsShape.push_back(lhsShape[i]);

    // The transpose output shape is what the dot was contracting against
    auto transposedShape =
        cast<RankedTensorType>(transposeOp.getType()).getShape();
    for (int64_t d : transposedShape)
      expandedLhsShape.push_back(d);
    for (int64_t i = contractDim + 1; i < lhsRank; ++i)
      expandedLhsShape.push_back(lhsShape[i]);

    auto expandedLhsType =
        RankedTensorType::get(expandedLhsShape, lhsType.getElementType());
    Value expandedLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, expandedLhsType, lhs);

    // Compute the inverse permutation, offset by batch-before dims,
    // with identity on batch-after dims
    int64_t numBatchBefore = contractDim;
    int64_t numBatchAfter = lhsRank - contractDim - 1;
    SmallVector<int64_t> invPerm(numBatchBefore + perm.size() + numBatchAfter);
    for (int64_t i = 0; i < numBatchBefore; ++i)
      invPerm[i] = i;
    // Inverse of perm: if perm[i] = j, then invPerm[j] = i
    SmallVector<int64_t> innerInvPerm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
      innerInvPerm[perm[i]] = i;
    for (size_t i = 0; i < innerInvPerm.size(); ++i)
      invPerm[numBatchBefore + i] = numBatchBefore + innerInvPerm[i];
    for (int64_t i = 0; i < numBatchAfter; ++i)
      invPerm[numBatchBefore + perm.size() + i] =
          numBatchBefore + perm.size() + i;

    // After inverse transpose, A's inner dims match X's shape (pre-transpose)
    SmallVector<int64_t> invTransposedShape;
    for (int64_t i = 0; i < numBatchBefore; ++i)
      invTransposedShape.push_back(expandedLhsShape[i]);
    for (int64_t d : inputShape)
      invTransposedShape.push_back(d);
    for (int64_t i = 0; i < numBatchAfter; ++i)
      invTransposedShape.push_back(
          expandedLhsShape[numBatchBefore + perm.size() + i]);

    auto invTransposedType =
        RankedTensorType::get(invTransposedShape, lhsType.getElementType());
    Value transposedLhs = stablehlo::TransposeOp::create(
        rewriter, loc, invTransposedType, expandedLhs,
        rewriter.getDenseI64ArrayAttr(invPerm));

    // Reshape back to [M, P]
    Value newLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, lhsType, transposedLhs);

    // Replace the RHS: remove the transpose, keep the reshape
    Value newRhs;
    if (reshapeOp) {
      auto reshapeType = cast<RankedTensorType>(reshapeOp.getType());
      newRhs = stablehlo::ReshapeOp::create(rewriter, loc, reshapeType,
                                            transposeInput);
    } else {
      newRhs = transposeInput;
    }

    auto resultType = cast<RankedTensorType>(dotOp.getType());
    auto newDot = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, newRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());

    rewriter.replaceOp(dotOp, newDot.getResult());
    patternApplied = true;
    return success();
  }
};

/// Match a channel-slice pattern: reshape(slice(source, ...)) where exactly one
/// dim is sliced to [channel:channel+1] and all others are full slices.
/// Returns (source, channelDim) or (nullptr, -1).
static std::pair<Value, int64_t> matchChannelSlice(Value val,
                                                   int64_t expectedChannel) {
  auto reshapeOp = val.getDefiningOp<stablehlo::ReshapeOp>();
  if (!reshapeOp)
    return {nullptr, -1};

  auto sliceOp = reshapeOp.getOperand().getDefiningOp<stablehlo::SliceOp>();
  if (!sliceOp)
    return {nullptr, -1};

  auto starts = sliceOp.getStartIndices();
  auto limits = sliceOp.getLimitIndices();
  auto stridesArr = sliceOp.getStrides();
  auto sourceType = cast<RankedTensorType>(sliceOp.getOperand().getType());
  auto sourceShape = sourceType.getShape();

  int64_t channelDim = -1;
  for (int64_t i = 0; i < sourceType.getRank(); ++i) {
    if (starts[i] == expectedChannel && limits[i] == expectedChannel + 1 &&
        stridesArr[i] == 1) {
      if (channelDim != -1)
        return {nullptr, -1}; // multiple channel dims
      channelDim = i;
    } else if (starts[i] == 0 && limits[i] == sourceShape[i] &&
               stridesArr[i] == 1) {
      // full slice — ok
    } else {
      return {nullptr, -1}; // partial non-channel slice
    }
  }

  if (channelDim == -1)
    return {nullptr, -1};
  return {sliceOp.getOperand(), channelDim};
}

/// DotAbsorbScatterHLO: Absorb a scatter-into-zeros from the RHS by gathering
/// columns from the LHS matrix.
/// dot_general(A:[M,N], scatter(zeros:[...], idx, data:[K]))
///   -> dot_general(gather(A, idx):[M,K], data)
/// where A and idx are sample-invariant.
///
/// Also handles the complex case:
/// Case 2: dot_general(A, complex(scatter(0,idx,re), scatter(0,idx,im)))
///   -> dot_general(gather(A,idx), complex(re,im))
/// Case 3: dot_general(A, reshape(complex(reshape(slice(X,[0:1,..])),
///                                         reshape(slice(X,[1:2,..])))))
///   where X = [optional transpose] + scatter_into_zeros with channel dim
///   -> dot_general(gather(A, linearized_idx), complex(re_updates, im_updates))
struct DotAbsorbScatterHLO : public OpRewritePattern<stablehlo::DotGeneralOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  DotAbsorbScatterHLO(SampleDependenceAnalysis &analysis, bool &patternApplied,
                      MLIRContext *context, PatternBenefit benefit = 5)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  /// Try to match a scatter-into-zeros pattern on a value.
  /// Returns (scatter_op, data) or (nullptr, nullptr).
  std::pair<stablehlo::ScatterOp, Value>
  matchScatterIntoZeros(Value val) const {
    auto scatterOp = val.getDefiningOp<stablehlo::ScatterOp>();
    if (!scatterOp || scatterOp.getInputs().size() != 1 ||
        scatterOp.getUpdates().size() != 1)
      return {nullptr, nullptr};

    // Check that input is a zero constant
    Value input = scatterOp.getInputs()[0];
    DenseElementsAttr constAttr;
    if (!matchPattern(input, m_Constant(&constAttr)) || !constAttr.isSplat() ||
        !constAttr.getSplatValue<APFloat>().isZero())
      return {nullptr, nullptr};

    // Check that update computation is just "return update" (overwrite)
    auto &updateRegion = scatterOp.getUpdateComputation();
    if (updateRegion.getBlocks().size() != 1)
      return {nullptr, nullptr};
    auto &block = updateRegion.front();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(block.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != block.getArgument(1))
      return {nullptr, nullptr};

    return {scatterOp, scatterOp.getUpdates()[0]};
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(dotOp))
      return failure();

    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();

    if (analysis.isSampleDependent(lhs))
      return failure();

    // Look through reshape on RHS
    auto [rhsInner, outerReshape] = lookThroughReshape(rhs);

    stablehlo::ScatterOp scatterOp;
    Value scatterIndices;
    Value newRhsData;

    // Case 1: Direct scatter
    auto [directScatter, directData] = matchScatterIntoZeros(rhsInner);
    if (directScatter) {
      scatterOp = directScatter;
      scatterIndices = directScatter.getScatterIndices();
      newRhsData = directData;
    }

    // Case 2: complex(scatter_re, scatter_im) — two separate scatters
    if (!scatterOp) {
      auto complexOp = rhsInner.getDefiningOp<stablehlo::ComplexOp>();
      if (complexOp) {
        auto [scatterRe, dataRe] = matchScatterIntoZeros(complexOp.getLhs());
        auto [scatterIm, dataIm] = matchScatterIntoZeros(complexOp.getRhs());
        if (scatterRe && scatterIm &&
            scatterRe.getScatterIndices() == scatterIm.getScatterIndices()) {
          scatterOp = scatterRe;
          scatterIndices = scatterRe.getScatterIndices();

          // Build complex(re_data, im_data)
          auto dataReType = cast<RankedTensorType>(dataRe.getType());
          auto complexElemType = ComplexType::get(dataReType.getElementType());
          auto complexDataType =
              RankedTensorType::get(dataReType.getShape(), complexElemType);
          newRhsData = stablehlo::ComplexOp::create(
              rewriter, loc, complexDataType, dataRe, dataIm);
        }
      }
    }

    // Case 3: complex(reshape(slice(X,[0:1,..])), reshape(slice(X,[1:2,..])))
    // where X = [optional transpose] + scatter_into_zeros with channel dim.
    // This handles NUFFT patterns where real/imag parts are stored as a channel
    // dimension in a single scatter, split via slicing.
    //
    // strideShape/strideDimMap: override the stride computation when a
    // transpose and channel dim are present.
    SmallVector<int64_t> strideShape;
    SmallVector<int64_t> strideDimMap;

    if (!scatterOp) {
      auto complexOp = rhsInner.getDefiningOp<stablehlo::ComplexOp>();
      if (!complexOp)
        return failure();

      auto [sourceRe, channelDimRe] = matchChannelSlice(complexOp.getLhs(), 0);
      auto [sourceIm, channelDimIm] = matchChannelSlice(complexOp.getRhs(), 1);
      if (!sourceRe || !sourceIm || sourceRe != sourceIm ||
          channelDimRe != channelDimIm)
        return failure();

      Value scatterInput = sourceRe;
      SmallVector<int64_t> transposePerm;
      int64_t channelDimInTransposed = channelDimRe;

      // Optional transpose between slice source and scatter
      if (auto transposeOp =
              scatterInput.getDefiningOp<stablehlo::TransposeOp>()) {
        transposePerm.assign(transposeOp.getPermutation().begin(),
                             transposeOp.getPermutation().end());
        scatterInput = transposeOp.getOperand();
      }

      auto [scatter, updates] = matchScatterIntoZeros(scatterInput);
      if (!scatter)
        return failure();

      // Verify channel dim structure:
      // update_window_dims should include exactly one dim (the channel dim)
      auto scatterDimNums = scatter.getScatterDimensionNumbers();
      auto updateWindowDims = scatterDimNums.getUpdateWindowDims();
      if (updateWindowDims.size() != 1)
        return failure();

      auto updatesType = cast<RankedTensorType>(updates.getType());
      auto updatesShape = updatesType.getShape();
      int64_t channelUpdateDim = updateWindowDims[0];
      int64_t channelSize = updatesShape[channelUpdateDim];
      if (channelSize != 2)
        return failure(); // must be real + imag
      if (updatesType.getRank() != 2)
        return failure(); // expect [channel, K] or [K, channel]

      int64_t K_local = updatesShape[1 - channelUpdateDim];

      scatterOp = scatter;
      scatterIndices = scatter.getScatterIndices();

      // Extract data: complex(updates[0,:], updates[1,:])
      // Slice updates along channel dim for re (channel 0) and im (channel 1)
      auto elemType = updatesType.getElementType();
      SmallVector<int64_t> reStart(2, 0),
          reLimit(updatesShape.begin(), updatesShape.end());
      reStart[channelUpdateDim] = 0;
      reLimit[channelUpdateDim] = 1;
      SmallVector<int64_t> imStart(2, 0),
          imLimit(updatesShape.begin(), updatesShape.end());
      imStart[channelUpdateDim] = 1;
      imLimit[channelUpdateDim] = 2;

      SmallVector<int64_t> slicedShape(updatesShape.begin(),
                                       updatesShape.end());
      slicedShape[channelUpdateDim] = 1;
      auto slicedType = RankedTensorType::get(slicedShape, elemType);
      SmallVector<int64_t> unitStrides = {1, 1};

      Value reSlice = stablehlo::SliceOp::create(
          rewriter, loc, slicedType, updates,
          rewriter.getDenseI64ArrayAttr(reStart),
          rewriter.getDenseI64ArrayAttr(reLimit),
          rewriter.getDenseI64ArrayAttr(unitStrides));
      Value imSlice = stablehlo::SliceOp::create(
          rewriter, loc, slicedType, updates,
          rewriter.getDenseI64ArrayAttr(imStart),
          rewriter.getDenseI64ArrayAttr(imLimit),
          rewriter.getDenseI64ArrayAttr(unitStrides));

      // Reshape slices to [K]
      auto flatType = RankedTensorType::get({K_local}, elemType);
      Value reData =
          stablehlo::ReshapeOp::create(rewriter, loc, flatType, reSlice);
      Value imData =
          stablehlo::ReshapeOp::create(rewriter, loc, flatType, imSlice);

      // Build complex(re, im) → [K, complex]
      auto complexElemType = ComplexType::get(elemType);
      auto complexDataType = RankedTensorType::get({K_local}, complexElemType);
      newRhsData = stablehlo::ComplexOp::create(rewriter, loc, complexDataType,
                                                reData, imData);

      // Compute effective stride shape and dim mapping.
      // The dot contracts over the spatial grid (scatter output with channel
      // dim removed, after transpose).
      auto scatterInputShapeRef =
          cast<RankedTensorType>(scatter.getInputs()[0].getType()).getShape();
      auto scatterDimsToOp = scatterDimNums.getScatterDimsToOperandDims();

      // Compute transposed shape and inverse perm
      SmallVector<int64_t> transposedShapeVec;
      SmallVector<int64_t> invPerm;
      if (!transposePerm.empty()) {
        transposedShapeVec.resize(transposePerm.size());
        invPerm.resize(transposePerm.size());
        for (size_t i = 0; i < transposePerm.size(); ++i) {
          transposedShapeVec[i] = scatterInputShapeRef[transposePerm[i]];
          invPerm[transposePerm[i]] = i;
        }
      } else {
        transposedShapeVec.assign(scatterInputShapeRef.begin(),
                                  scatterInputShapeRef.end());
        invPerm.resize(scatterInputShapeRef.size());
        for (size_t i = 0; i < invPerm.size(); ++i)
          invPerm[i] = i;
      }

      // Remove channel dim from transposed shape to get spatial shape
      for (size_t i = 0; i < transposedShapeVec.size(); ++i) {
        if ((int64_t)i != channelDimInTransposed)
          strideShape.push_back(transposedShapeVec[i]);
      }

      // Map each scatter index column to a spatial stride dim:
      // index col d → scatter operand dim → transposed dim (via invPerm)
      // → spatial dim (adjust for channel removal)
      for (int64_t d = 0; d < (int64_t)scatterDimsToOp.size(); ++d) {
        int64_t operandDim = scatterDimsToOp[d];
        int64_t transposedDim = invPerm[operandDim];
        int64_t spatialDim = transposedDim > channelDimInTransposed
                                 ? transposedDim - 1
                                 : transposedDim;
        strideDimMap.push_back(spatialDim);
      }
    }

    if (!scatterOp)
      return failure();

    // Scatter indices must be sample-invariant
    if (analysis.isSampleDependent(scatterIndices))
      return failure();

    // Get scatter dimension numbers to understand the index structure
    auto scatterDimNumbers = scatterOp.getScatterDimensionNumbers();
    auto scatterDimsToOperand = scatterDimNumbers.getScatterDimsToOperandDims();

    // Get the scatter input shape (the zeros tensor shape)
    auto scatterInputType =
        cast<RankedTensorType>(scatterOp.getInputs()[0].getType());
    auto scatterInputShape = scatterInputType.getShape();

    // Build linear indices from the scatter indices.
    // scatter_indices has shape [K, num_index_dims] (when indexVectorDim=1).
    auto indicesType = cast<RankedTensorType>(scatterIndices.getType());
    auto indicesShape = indicesType.getShape();
    int64_t K = indicesShape[0]; // number of scatter points

    // For Case 3, strideShape was already computed from the transposed
    // spatial dims. For Cases 1&2, use scatterInputShape directly.
    if (strideShape.empty()) {
      strideShape.assign(scatterInputShape.begin(), scatterInputShape.end());
      for (int64_t d : scatterDimsToOperand)
        strideDimMap.push_back(d);
    }

    // Compute the total size of the effective grid (= P for the dot)
    int64_t scatterGridSize = 1;
    for (int64_t d : strideShape)
      scatterGridSize *= d;

    // We need to linearize the multi-dim scatter indices into 1D indices
    // so we can gather columns from A (which has P = scatterGridSize)
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsShape = lhsType.getShape();

    // Handle single contracting dim on LHS (first or last dim)
    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();
    if (lhsContractDims.size() != 1)
      return failure();
    int64_t contractDim = lhsContractDims[0];
    if (contractDim != 0 && contractDim != lhsType.getRank() - 1)
      return failure();

    int64_t P = lhsShape[contractDim];
    if (P != scatterGridSize)
      return failure();

    // Linearize scatter indices using effective strides.
    // linear_idx = sum_d idx[:,d] * strides[strideDimMap[d]]
    int64_t numIndexDims = scatterDimsToOperand.size();

    // Compute strides from strideShape (row-major)
    SmallVector<int64_t> strides(strideShape.size());
    int64_t stride = 1;
    for (int64_t i = strideShape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= strideShape[i];
    }

    auto i64Type = rewriter.getI64Type();
    auto linearIdxType = RankedTensorType::get({K, 1}, i64Type);

    // Start with zeros
    Value linearIdx = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(linearIdxType, rewriter.getI64IntegerAttr(0)));

    for (int64_t d = 0; d < numIndexDims; ++d) {
      // Extract idx[:, d]
      auto sliceType =
          RankedTensorType::get({K, 1}, indicesType.getElementType());
      Value idxSlice =
          stablehlo::SliceOp::create(rewriter, loc, sliceType, scatterIndices,
                                     rewriter.getDenseI64ArrayAttr({0, d}),
                                     rewriter.getDenseI64ArrayAttr({K, d + 1}),
                                     rewriter.getDenseI64ArrayAttr({1, 1}));

      // Convert to i64 if needed
      if (indicesType.getElementType() != i64Type) {
        idxSlice = stablehlo::ConvertOp::create(
            rewriter, loc, RankedTensorType::get({K, 1}, i64Type), idxSlice);
      }

      // Multiply by stride (using strideDimMap for the effective dim)
      int64_t effectiveDim = strideDimMap[d];
      Value strideVal = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(linearIdxType, rewriter.getI64IntegerAttr(
                                                    strides[effectiveDim])));
      Value scaled = stablehlo::MulOp::create(rewriter, loc, linearIdxType,
                                              idxSlice, strideVal);
      linearIdx = stablehlo::AddOp::create(rewriter, loc, linearIdxType,
                                           linearIdx, scaled);
    }

    // Gather columns from A using the linear indices.
    // A is [batchBefore..., P, batchAfter...], gather to
    // [batchBefore..., K, batchAfter...].
    // In the result, K goes at the contractDim position and offset dims
    // (all non-contracted dims) fill their original positions.
    SmallVector<int64_t> offsetDims;
    for (int64_t i = 0; i < contractDim; ++i)
      offsetDims.push_back(i);
    for (int64_t i = contractDim + 1; i < lhsType.getRank(); ++i)
      offsetDims.push_back(i);
    SmallVector<int64_t> collapsedSliceDims = {contractDim};
    SmallVector<int64_t> startIndexMap = {contractDim};

    SmallVector<int64_t> sliceSizes;
    for (int64_t i = 0; i < lhsType.getRank(); ++i)
      sliceSizes.push_back(i == contractDim ? 1 : lhsShape[i]);

    SmallVector<int64_t> gatheredShape;
    for (int64_t i = 0; i < contractDim; ++i)
      gatheredShape.push_back(lhsShape[i]);
    gatheredShape.push_back(K);
    for (int64_t i = contractDim + 1; i < lhsType.getRank(); ++i)
      gatheredShape.push_back(lhsShape[i]);

    auto gatheredType =
        RankedTensorType::get(gatheredShape, lhsType.getElementType());
    auto gatherDimNumbers = stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(), offsetDims, collapsedSliceDims,
        /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
        startIndexMap, /*indexVectorDim=*/1);

    Value gatheredLhs = stablehlo::GatherOp::create(
        rewriter, loc, gatheredType, lhs, linearIdx, gatherDimNumbers,
        rewriter.getDenseI64ArrayAttr(sliceSizes));

    // Build new RHS from data
    // The data shape should be [K, ...window_dims...]
    // We may need to reshape it to match the dot's expected RHS shape
    auto dataType = cast<RankedTensorType>(newRhsData.getType());
    Value finalRhs = newRhsData;

    // If there was an outer reshape, create a matching reshape for the new data
    if (outerReshape) {
      // The original RHS was reshape(scatter_result) -> [P, trailing...]
      // New RHS should be reshape(data) -> [K, trailing...]
      auto outerReshapeType = cast<RankedTensorType>(outerReshape.getType());
      auto outerShape = outerReshapeType.getShape();

      // Replace P with K in the reshape shape
      SmallVector<int64_t> newRhsShape;
      newRhsShape.push_back(K);
      for (size_t i = 1; i < outerShape.size(); ++i)
        newRhsShape.push_back(outerShape[i]);

      auto newRhsType =
          RankedTensorType::get(newRhsShape, dataType.getElementType());
      finalRhs =
          stablehlo::ReshapeOp::create(rewriter, loc, newRhsType, newRhsData);
    }

    // Build new dot_general with contracted dim size K instead of P
    auto resultType = cast<RankedTensorType>(dotOp.getType());
    auto newDot = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, gatheredLhs, finalRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());

    rewriter.replaceOp(dotOp, newDot.getResult());
    patternApplied = true;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Element-wise scale factorization patterns
//===----------------------------------------------------------------------===//

/// PowerScaleFactorizationHLO:
/// pow(broadcast(s) * A, N) -> broadcast(pow(s, N)) * pow(A, N)
/// where A is sample-invariant.
struct PowerScaleFactorizationHLO : public OpRewritePattern<stablehlo::PowOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  PowerScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                             bool &patternApplied, MLIRContext *context,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::PowOp powOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(powOp))
      return failure();

    Location loc = powOp.getLoc();
    Value base = powOp.getLhs();
    Value exponent = powOp.getRhs();

    auto [scale, invariant] = matchScaledInvariant(base, analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(powOp.getType());

    // pow(A, N) — sample-invariant, will be hoisted
    auto powInvariant = stablehlo::PowOp::create(rewriter, loc, resultType,
                                                 invariant, exponent);

    // pow(broadcast(scale), N) — broadcast scale to full shape, then pow
    Value broadcastedScale =
        broadcastScalarToShape(rewriter, loc, scale, resultType);
    auto powScale = stablehlo::PowOp::create(rewriter, loc, resultType,
                                             broadcastedScale, exponent);

    // pow(s,N) * pow(A,N)
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType, powScale,
                                           powInvariant);

    rewriter.replaceOp(powOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// SqrtScaleFactorizationHLO:
/// sqrt(broadcast(s) * A) -> broadcast(sqrt(s)) * sqrt(A)
/// where A is sample-invariant.
struct SqrtScaleFactorizationHLO : public OpRewritePattern<stablehlo::SqrtOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  SqrtScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                            bool &patternApplied, MLIRContext *context,
                            PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::SqrtOp sqrtOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(sqrtOp))
      return failure();

    Location loc = sqrtOp.getLoc();
    auto [scale, invariant] =
        matchScaledInvariant(sqrtOp.getOperand(), analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(sqrtOp.getType());

    // sqrt(A) — invariant
    auto sqrtInvariant =
        stablehlo::SqrtOp::create(rewriter, loc, resultType, invariant);

    // sqrt(scale)
    Value scalarScale = scale;
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto sqrtScale = stablehlo::SqrtOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);

    Value broadcastedSqrt =
        broadcastScalarToShape(rewriter, loc, sqrtScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedSqrt, sqrtInvariant);

    rewriter.replaceOp(sqrtOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// RsqrtScaleFactorizationHLO:
/// rsqrt(broadcast(s) * A) -> broadcast(rsqrt(s)) * rsqrt(A)
/// where A is sample-invariant.
struct RsqrtScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::RsqrtOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  RsqrtScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                             bool &patternApplied, MLIRContext *context,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::RsqrtOp rsqrtOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(rsqrtOp))
      return failure();

    Location loc = rsqrtOp.getLoc();
    auto [scale, invariant] =
        matchScaledInvariant(rsqrtOp.getOperand(), analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(rsqrtOp.getType());

    auto rsqrtInvariant =
        stablehlo::RsqrtOp::create(rewriter, loc, resultType, invariant);

    Value scalarScale = scale;
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto rsqrtScale = stablehlo::RsqrtOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);

    Value broadcastedRsqrt =
        broadcastScalarToShape(rewriter, loc, rsqrtScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedRsqrt, rsqrtInvariant);

    rewriter.replaceOp(rsqrtOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// AbsScaleFactorizationHLO:
/// abs(broadcast(s) * A) -> broadcast(abs(s)) * abs(A)
/// where A is sample-invariant.
struct AbsScaleFactorizationHLO : public OpRewritePattern<stablehlo::AbsOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  AbsScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                           bool &patternApplied, MLIRContext *context,
                           PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::AbsOp absOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(absOp))
      return failure();

    Location loc = absOp.getLoc();
    auto [scale, invariant] =
        matchScaledInvariant(absOp.getOperand(), analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(absOp.getType());
    auto inputType = cast<RankedTensorType>(absOp.getOperand().getType());

    // abs(A) — invariant
    // Note: abs may change the element type (complex -> real)
    auto absInvariantType = RankedTensorType::get(inputType.getShape(),
                                                  resultType.getElementType());
    auto absInvariant =
        stablehlo::AbsOp::create(rewriter, loc, absInvariantType, invariant);

    // abs(scale)
    Value scalarScale = scale;
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto absScaleType = RankedTensorType::get({}, resultType.getElementType());
    auto absScale =
        stablehlo::AbsOp::create(rewriter, loc, absScaleType, scalarScale);

    Value broadcastedAbs =
        broadcastScalarToShape(rewriter, loc, absScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedAbs, absInvariant);

    rewriter.replaceOp(absOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// NegateDistributionHLO:
/// negate(broadcast(s) * A) -> broadcast(negate(s)) * A
/// where A is sample-invariant.
struct NegateDistributionHLO : public OpRewritePattern<stablehlo::NegOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  NegateDistributionHLO(SampleDependenceAnalysis &analysis,
                        bool &patternApplied, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::NegOp negOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(negOp))
      return failure();

    Location loc = negOp.getLoc();
    auto [scale, invariant] =
        matchScaledInvariant(negOp.getOperand(), analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(negOp.getType());

    // negate(scale)
    Value scalarScale = scale;
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto negScale = stablehlo::NegOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);

    // broadcast(-s) * A — A is already invariant
    Value broadcastedNeg =
        broadcastScalarToShape(rewriter, loc, negScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedNeg, invariant);

    rewriter.replaceOp(negOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// Try to decompose a value as broadcast(scalar) + invariant_matrix.
/// Returns (scalar, invariant) on success, (nullptr, nullptr) on failure.
static std::pair<Value, Value>
matchAddedInvariant(Value value, SampleDependenceAnalysis &analysis) {
  auto addOp = value.getDefiningOp<stablehlo::AddOp>();
  if (!addOp)
    return {nullptr, nullptr};

  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();

  auto lhsBroadcast = lhs.getDefiningOp<stablehlo::BroadcastInDimOp>();
  auto rhsBroadcast = rhs.getDefiningOp<stablehlo::BroadcastInDimOp>();

  Value scale, invariant;
  if (lhsBroadcast && isScalar(lhsBroadcast.getOperand())) {
    scale = lhsBroadcast.getOperand();
    invariant = rhs;
  } else if (rhsBroadcast && isScalar(rhsBroadcast.getOperand())) {
    scale = rhsBroadcast.getOperand();
    invariant = lhs;
  } else if (isScalar(lhs) && !isScalar(rhs)) {
    scale = lhs;
    invariant = rhs;
  } else if (isScalar(rhs) && !isScalar(lhs)) {
    scale = rhs;
    invariant = lhs;
  } else {
    return {nullptr, nullptr};
  }

  if (analysis.isSampleDependent(invariant))
    return {nullptr, nullptr};

  return {scale, invariant};
}

/// ExpAddFactorizationHLO:
/// exp(broadcast(s) + A) -> broadcast(exp(s)) * exp(A)
/// where A is sample-invariant.
struct ExpAddFactorizationHLO : public OpRewritePattern<stablehlo::ExpOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  ExpAddFactorizationHLO(SampleDependenceAnalysis &analysis,
                         bool &patternApplied, MLIRContext *context,
                         PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::ExpOp expOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(expOp))
      return failure();

    Location loc = expOp.getLoc();
    auto [scale, invariant] = matchAddedInvariant(expOp.getOperand(), analysis);
    if (!scale)
      return failure();

    auto resultType = cast<RankedTensorType>(expOp.getType());

    // exp(A) — invariant
    auto expInvariant =
        stablehlo::ExpOp::create(rewriter, loc, resultType, invariant);

    // exp(scale)
    Value scalarScale = scale;
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getRank() > 0) {
      auto scalarType = RankedTensorType::get({}, scaleType.getElementType());
      scalarScale =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, scale);
    }
    auto expScale = stablehlo::ExpOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);

    // broadcast(exp(s)) * exp(A)
    Value broadcastedExp =
        broadcastScalarToShape(rewriter, loc, expScale, resultType);
    auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                           broadcastedExp, expInvariant);

    rewriter.replaceOp(expOp, result.getResult());
    patternApplied = true;
    return success();
  }
};

/// DivideScaleFactorizationHLO:
/// divide(broadcast(s) * A, B) -> broadcast(s) * divide(A, B)
///   when A, B are sample-invariant (divide becomes hoistable)
/// divide(X, broadcast(s) * A) -> broadcast(1/s) * divide(X, A)
///   when A is sample-invariant
struct DivideScaleFactorizationHLO : public OpRewritePattern<stablehlo::DivOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  DivideScaleFactorizationHLO(SampleDependenceAnalysis &analysis,
                              bool &patternApplied, MLIRContext *context,
                              PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::DivOp divOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(divOp))
      return failure();

    Location loc = divOp.getLoc();
    Value numerator = divOp.getLhs();
    Value denominator = divOp.getRhs();
    auto resultType = cast<RankedTensorType>(divOp.getType());

    // Case 1: numerator = broadcast(s) * A, both A and denominator invariant
    auto [numScale, numInvariant] = matchScaledInvariant(numerator, analysis);
    if (numScale && !analysis.isSampleDependent(denominator)) {
      // divide(A, B) — fully invariant
      auto divInvariant = stablehlo::DivOp::create(rewriter, loc, resultType,
                                                   numInvariant, denominator);

      Value broadcastedScale =
          broadcastScalarToShape(rewriter, loc, numScale, resultType);
      auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                             broadcastedScale, divInvariant);

      rewriter.replaceOp(divOp, result.getResult());
      patternApplied = true;
      return success();
    }

    // Case 2: denominator = broadcast(s) * A, A invariant
    auto [denScale, denInvariant] = matchScaledInvariant(denominator, analysis);
    if (denScale) {
      // divide(X, A) — A invariant, may or may not be fully hoistable
      auto newDiv = stablehlo::DivOp::create(rewriter, loc, resultType,
                                             numerator, denInvariant);

      // Compute 1/s
      auto scaleType = cast<RankedTensorType>(denScale.getType());
      auto elemType = scaleType.getElementType();
      auto scalarType = RankedTensorType::get({}, elemType);
      Value one = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(scalarType,
                                 rewriter.getFloatAttr(elemType, 1.0)));
      Value scalarScale = denScale;
      if (scaleType.getRank() > 0)
        scalarScale =
            stablehlo::ReshapeOp::create(rewriter, loc, scalarType, denScale);
      Value invScale =
          stablehlo::DivOp::create(rewriter, loc, scalarType, one, scalarScale);

      Value broadcastedInv =
          broadcastScalarToShape(rewriter, loc, invScale, resultType);
      auto result = stablehlo::MulOp::create(rewriter, loc, resultType,
                                             broadcastedInv, newDiv);

      rewriter.replaceOp(divOp, result.getResult());
      patternApplied = true;
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Eigendecomposition lift for diagonal additive perturbation
//===----------------------------------------------------------------------===//

/// Check if a value is a constant identity matrix (1.0 on diagonal, 0.0
/// elsewhere).
static bool isIdentityMatrix(Value value) {
  auto constOp = value.getDefiningOp<stablehlo::ConstantOp>();
  if (!constOp)
    return false;
  auto attr = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!attr)
    return false;
  auto type = cast<RankedTensorType>(attr.getType());
  if (type.getRank() != 2 || type.getShape()[0] != type.getShape()[1])
    return false;
  int64_t n = type.getShape()[0];
  for (auto [idx, val] : llvm::enumerate(attr.getValues<APFloat>())) {
    int64_t row = idx / n, col = idx % n;
    bool shouldBeOne = (row == col);
    if (shouldBeOne && !val.isExactlyValue(1.0))
      return false;
    if (!shouldBeOne && !val.isExactlyValue(0.0))
      return false;
  }
  return true;
}

/// Try to decompose `multiply(broadcast(scalar), matrix)` into (scalar,
/// matrix). Returns (nullptr, nullptr) if the pattern doesn't match.
static std::pair<Value, Value> decomposeScaledMatrix(Value value) {
  auto mulOp = value.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp)
    return {nullptr, nullptr};

  for (auto [maybeScalarBcast, maybeMatrix] :
       {std::pair{mulOp.getLhs(), mulOp.getRhs()},
        std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
    auto bcast = maybeScalarBcast.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (bcast && isScalar(bcast.getOperand()))
      return {bcast.getOperand(), maybeMatrix};
    if (isScalar(maybeScalarBcast))
      return {maybeScalarBcast, maybeMatrix};
  }
  return {nullptr, nullptr};
}

/// Try to decompose a value as s*A + t*I where A is sample-invariant.
/// Returns (invariantMatrix, scaleScalar, diagScalar) on success.
/// Also handles the simpler A + t*I case (scaleScalar is nullptr).
struct ScaledInvariantPlusDiag {
  Value invariant;       // The sample-invariant matrix A
  Value scaleScalar;     // s (nullptr if pattern is A + t*I or A + t*B)
  Value diagScalar;      // t
  Value secondInvariant; // B (nullptr for identity case: s*A + t*I)
};

static ScaledInvariantPlusDiag
matchScaledInvariantPlusDiagShift(Value value,
                                  SampleDependenceAnalysis &analysis) {
  auto addOp = value.getDefiningOp<stablehlo::AddOp>();
  if (!addOp)
    return {nullptr, nullptr, nullptr, nullptr};

  // Try both orderings of the add operands
  for (auto [lhs, rhs] : {std::pair{addOp.getLhs(), addOp.getRhs()},
                          std::pair{addOp.getRhs(), addOp.getLhs()}}) {
    // rhs should be t*I or t*B where B is sample-invariant
    Value diagScalar = nullptr;
    Value secondInvariant = nullptr;
    auto [dScalar, dMatrix] = decomposeScaledMatrix(rhs);
    if (dScalar) {
      if (isIdentityMatrix(dMatrix)) {
        // Identity case: rhs = t*I
        diagScalar = dScalar;
      } else if (!analysis.isSampleDependent(dMatrix)) {
        // Generalized case: rhs = t*B where B is invariant
        diagScalar = dScalar;
        secondInvariant = dMatrix;
      }
    }
    if (!diagScalar)
      continue;

    // lhs is either: (a) invariant matrix A, or (b) s*A where A is invariant
    if (!analysis.isSampleDependent(lhs)) {
      // Simple case: A + t*I or A + t*B
      return {lhs, nullptr, diagScalar, secondInvariant};
    }

    // Try decomposing lhs as s*A
    auto [sScalar, sMatrix] = decomposeScaledMatrix(lhs);
    if (sScalar && !analysis.isSampleDependent(sMatrix)) {
      // Scaled case: s*A + t*I or s*A + t*B
      return {sMatrix, sScalar, diagScalar, secondInvariant};
    }
  }
  return {nullptr, nullptr, nullptr, nullptr};
}

/// Detect chained triangular_solve pairs that compute Σ^{-1}b.
///
/// Pattern: triangular_solve(F, triangular_solve(F, b, ADJOINT), NO_TRANSPOSE)
/// where F is a Cholesky factor (upper or lower). The outer solve's `b` operand
/// IS the inner solve result.
///
/// Returns the ultimate RHS `b` if the pattern matches, nullptr otherwise.
static Value matchFullCholeskySolve(stablehlo::TriangularSolveOp outerSolve,
                                    Value cholResult) {
  // The outer solve must use Cholesky result directly, NO_TRANSPOSE
  if (outerSolve.getA() != cholResult)
    return nullptr;
  if (!outerSolve.getLeftSide())
    return nullptr;
  if (outerSolve.getTransposeA() != stablehlo::Transpose::NO_TRANSPOSE)
    return nullptr;

  // The outer solve's B must come from an inner triangular_solve using the
  // same Cholesky, with ADJOINT transpose
  auto innerSolve =
      outerSolve.getB().getDefiningOp<stablehlo::TriangularSolveOp>();
  if (!innerSolve)
    return nullptr;
  if (innerSolve.getA() != cholResult)
    return nullptr;
  if (!innerSolve.getLeftSide())
    return nullptr;
  if (innerSolve.getTransposeA() != stablehlo::Transpose::ADJOINT)
    return nullptr;

  return innerSolve.getB();
}

/// Detect diagonal extraction: multiply(cholResult, identityMatrix) or
/// dot_general(cholResult, identity, batching=[1]x[1], contracting=[0]x[0])
/// used to get diagonal elements for log-determinant.
static bool isDiagonalExtraction(Operation *user, Value cholResult) {
  // Pattern 1: element-wise multiply(L, I)
  if (auto mulOp = dyn_cast<stablehlo::MulOp>(user)) {
    for (auto [a, b] : {std::pair{mulOp.getLhs(), mulOp.getRhs()},
                        std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
      if (a == cholResult && isIdentityMatrix(b))
        return true;
    }
    return false;
  }
  // Pattern 2: dot_general(L, I, batching=[1]x[1], contracting=[0]x[0])
  // This is the pattern Reactant produces for extracting diag(L).
  if (auto dotOp = dyn_cast<stablehlo::DotGeneralOp>(user)) {
    auto dims = dotOp.getDotDimensionNumbers();
    // Check batching_dims = [1] x [1], contracting_dims = [0] x [0]
    if (dims.getLhsBatchingDimensions().size() != 1 ||
        dims.getRhsBatchingDimensions().size() != 1 ||
        dims.getLhsContractingDimensions().size() != 1 ||
        dims.getRhsContractingDimensions().size() != 1)
      return false;
    if (dims.getLhsBatchingDimensions()[0] != 1 ||
        dims.getRhsBatchingDimensions()[0] != 1 ||
        dims.getLhsContractingDimensions()[0] != 0 ||
        dims.getRhsContractingDimensions()[0] != 0)
      return false;
    for (auto [a, b] : {std::pair{dotOp.getLhs(), dotOp.getRhs()},
                        std::pair{dotOp.getRhs(), dotOp.getLhs()}}) {
      if (a == cholResult && isIdentityMatrix(b))
        return true;
    }
  }
  return false;
}

/// Eigendecomposition lift for Cholesky with diagonal additive perturbation.
///
/// Rewrites cholesky(s*A + t*I) where A is sample-invariant, replacing:
///   - Full solve chains: F\(F'\b) → Q @ diag(1/(s*λ+t)) @ Q^T @ b
///   - Single solves: F\b (NO_TRANSPOSE) → Q @ diag(1/sqrt(s*λ+t)) @ Q^T @ b
///                    F'\b (ADJOINT)     → Q @ diag(1/sqrt(s*λ+t)) @ Q^T @ b
///   - Diagonal extraction: diag(F) → sqrt(s*λ+t) (for log-determinant)
///
/// Also handles the simpler case: cholesky(A + t*I) (s=1 implicitly).
///
/// The eigendecomposition enzymexla.lapack.syevd(A) is sample-invariant and
/// will be automatically hoisted out of the mcmc_region in Phase 2.
struct CholeskyEigenLiftHLO : public OpRewritePattern<stablehlo::CholeskyOp> {
  SampleDependenceAnalysis &analysis;
  bool &patternApplied;

  CholeskyEigenLiftHLO(SampleDependenceAnalysis &analysis, bool &patternApplied,
                       MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), analysis(analysis),
        patternApplied(patternApplied) {}

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp cholOp,
                                PatternRewriter &rewriter) const override {
    if (!analysis.isInTargetRegion(cholOp))
      return failure();

    Location loc = cholOp.getLoc();

    // 1. Match input = s*A + t*I (or A + t*I)
    auto match = matchScaledInvariantPlusDiagShift(cholOp.getA(), analysis);
    if (!match.invariant)
      return failure();

    Value cholResult = cholOp.getResult();

    // 2. Classify ALL users into supported patterns
    // - Full solve chains: outer_solve(F, inner_solve(F^H, b)) → Σ^{-1}b
    // - Single triangular_solve(F, b)
    // - Diagonal extraction: multiply(F, I) for log-determinant
    struct FullSolveInfo {
      stablehlo::TriangularSolveOp outerSolve;
      stablehlo::TriangularSolveOp innerSolve;
      Value rhs; // The ultimate b in Σ^{-1}b
    };
    SmallVector<FullSolveInfo> fullSolves;
    struct SingleSolveInfo {
      stablehlo::TriangularSolveOp solve;
      bool isAdjoint; // ADJOINT vs NO_TRANSPOSE
    };
    SmallVector<SingleSolveInfo> singleSolves;
    SmallVector<Operation *> diagExtractions;

    // First pass: identify full solve chains (so we don't double-count inner
    // solves)
    DenseSet<Operation *> innerSolveOps;

    for (auto *user : cholResult.getUsers()) {
      auto triSolve = dyn_cast<stablehlo::TriangularSolveOp>(user);
      if (!triSolve) {
        if (isDiagonalExtraction(user, cholResult)) {
          diagExtractions.push_back(user);
          continue;
        }
        // Skip unsupported users — the cholesky won't be erased but
        // recognized users (solves, diag extractions) can still be replaced.
        continue;
      }

      if (!triSolve.getLeftSide())
        return failure();

      // Check if this is an outer solve of a full chain
      if (triSolve.getTransposeA() == stablehlo::Transpose::NO_TRANSPOSE) {
        Value rhs = matchFullCholeskySolve(triSolve, cholResult);
        if (rhs) {
          auto innerSolve =
              triSolve.getB().getDefiningOp<stablehlo::TriangularSolveOp>();
          fullSolves.push_back({triSolve, innerSolve, rhs});
          innerSolveOps.insert(innerSolve.getOperation());
          continue;
        }
      }
    }

    // Second pass: remaining triangular_solves are single solves
    for (auto *user : cholResult.getUsers()) {
      auto triSolve = dyn_cast<stablehlo::TriangularSolveOp>(user);
      if (!triSolve)
        continue;
      if (innerSolveOps.count(triSolve.getOperation()))
        continue;
      // Check it wasn't already classified as a full solve outer
      bool isFullSolveOuter = false;
      for (auto &fs : fullSolves) {
        if (fs.outerSolve == triSolve) {
          isFullSolveOuter = true;
          break;
        }
      }
      if (isFullSolveOuter)
        continue;

      auto transpose = triSolve.getTransposeA();
      if (transpose == stablehlo::Transpose::NO_TRANSPOSE) {
        singleSolves.push_back({triSolve, /*isAdjoint=*/false});
      } else if (transpose == stablehlo::Transpose::ADJOINT) {
        singleSolves.push_back({triSolve, /*isAdjoint=*/true});
      } else {
        return failure(); // Unsupported transpose mode
      }
    }

    // Must have at least one solvable user
    if (fullSolves.empty() && singleSolves.empty() && diagExtractions.empty())
      return failure();

    auto inputType = cast<RankedTensorType>(match.invariant.getType());
    int64_t N = inputType.getShape()[0];
    auto elemType = inputType.getElementType();
    auto eigvalsType = RankedTensorType::get({N}, elemType);
    auto infoType = RankedTensorType::get({}, rewriter.getIntegerType(64));

    // --- Generalized case: whiten through chol(B) ---
    // When match.secondInvariant is set, we have chol(s*A + t*B) where both
    // A, B are invariant. We whiten: L_B = chol(B), C_w = L_B^{-1}*A*L_B^{-T},
    // then eigendecompose C_w instead of A directly.
    Value LB;
    Value eigenTarget = match.invariant;
    if (match.secondInvariant) {
      // L_B = chol(B) — invariant, will be hoisted
      LB = stablehlo::CholeskyOp::create(rewriter, loc, inputType,
                                         match.secondInvariant,
                                         rewriter.getBoolAttr(true))
               .getResult();

      // Y = L_B^{-1} * A  (solve L_B * Y = A)
      Value Y = stablehlo::TriangularSolveOp::create(
                    rewriter, loc, inputType, LB, match.invariant,
                    /*left_side=*/true, /*lower=*/true,
                    /*unit_diagonal=*/false, stablehlo::Transpose::NO_TRANSPOSE)
                    .getResult();

      // C_w = Y * L_B^{-T}  (solve X * L_B^T = Y, i.e., left_side=false)
      eigenTarget = stablehlo::TriangularSolveOp::create(
                        rewriter, loc, inputType, LB, Y,
                        /*left_side=*/false, /*lower=*/true,
                        /*unit_diagonal=*/false, stablehlo::Transpose::ADJOINT)
                        .getResult();
    }

    // 3. Create eigendecomposition (sample-invariant, will be auto-hoisted)
    auto syevd = enzymexla::SyevdOp::create(
        rewriter, loc, TypeRange{inputType, eigvalsType, infoType}, eigenTarget,
        enzymexla::LapackUploAttr::get(rewriter.getContext(),
                                       enzymexla::LapackUplo::L));
    Value Q = syevd.getEigenvectors();
    Value lambda = syevd.getEigenvalues();

    // 4. Compute shifted eigenvalues: d = s*lambda + t (or lambda + t)
    Value shifted;
    if (match.scaleScalar) {
      Value sBcast =
          broadcastScalarToShape(rewriter, loc, match.scaleScalar, eigvalsType);
      Value sLambda =
          stablehlo::MulOp::create(rewriter, loc, eigvalsType, sBcast, lambda);
      Value tBcast =
          broadcastScalarToShape(rewriter, loc, match.diagScalar, eigvalsType);
      shifted =
          stablehlo::AddOp::create(rewriter, loc, eigvalsType, sLambda, tBcast);
    } else {
      Value tBcast =
          broadcastScalarToShape(rewriter, loc, match.diagScalar, eigvalsType);
      shifted =
          stablehlo::AddOp::create(rewriter, loc, eigvalsType, lambda, tBcast);
    }

    Value ones = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(eigvalsType,
                               rewriter.getFloatAttr(elemType, 1.0)));

    // Precompute Q^T
    Value QT = stablehlo::TransposeOp::create(
        rewriter, loc, inputType, Q, rewriter.getDenseI64ArrayAttr({1, 0}));

    auto dotDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhsBatchingDims=*/{}, /*rhsBatchingDims=*/{},
        /*lhsContractingDims=*/{1}, /*rhsContractingDims=*/{0});

    // Helper: compute Q @ diag(scale) @ Q^T @ b
    auto emitEigenSolve = [&](Value b, Value scaleVec,
                              RankedTensorType resultType) -> Value {
      Value QTb = stablehlo::DotGeneralOp::create(
          rewriter, loc, resultType, QT, b, dotDims,
          /*precision_config=*/nullptr, /*algorithm=*/nullptr);
      Value scaleBcast = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, resultType, scaleVec,
          rewriter.getDenseI64ArrayAttr({0}));
      Value scaled =
          stablehlo::MulOp::create(rewriter, loc, resultType, scaleBcast, QTb);
      return stablehlo::DotGeneralOp::create(
          rewriter, loc, resultType, Q, scaled, dotDims,
          /*precision_config=*/nullptr, /*algorithm=*/nullptr);
    };

    // 5. Replace full solve chains: Σ^{-1}b
    //   Identity case: Q @ diag(1/d) @ Q^T @ b
    //   Generalized:   L_B^{-T} @ Q @ diag(1/d) @ Q^T @ L_B^{-1} @ b
    if (!fullSolves.empty()) {
      Value invShifted =
          stablehlo::DivOp::create(rewriter, loc, eigvalsType, ones, shifted);
      for (auto &fs : fullSolves) {
        rewriter.setInsertionPoint(fs.outerSolve);
        auto resultType = cast<RankedTensorType>(fs.outerSolve.getType());
        Value rhs = fs.rhs;

        if (LB) {
          // z1 = L_B^{-1} @ b
          rhs = stablehlo::TriangularSolveOp::create(
                    rewriter, loc, resultType, LB, rhs,
                    /*left_side=*/true, /*lower=*/true,
                    /*unit_diagonal=*/false, stablehlo::Transpose::NO_TRANSPOSE)
                    .getResult();
        }

        Value result = emitEigenSolve(rhs, invShifted, resultType);

        if (LB) {
          // result = L_B^{-T} @ result
          result = stablehlo::TriangularSolveOp::create(
                       rewriter, loc, resultType, LB, result,
                       /*left_side=*/true, /*lower=*/true,
                       /*unit_diagonal=*/false, stablehlo::Transpose::ADJOINT)
                       .getResult();
        }

        rewriter.replaceOp(fs.outerSolve, result);
        if (fs.innerSolve->use_empty())
          rewriter.eraseOp(fs.innerSolve);
      }
    }

    // 6. Replace single solves: L^{-1}b or L^{-H}b
    //   Identity case:   Q @ diag(1/sqrt(d)) @ Q^T @ b
    //   Generalized:     Q @ diag(1/sqrt(d)) @ Q^T @ L_B^{-1} @ b
    //   Both preserve |L^{-1}b|^2 = b^T K^{-1} b.
    if (!singleSolves.empty()) {
      Value sqrtShifted =
          stablehlo::SqrtOp::create(rewriter, loc, eigvalsType, shifted);
      Value invSqrtShifted = stablehlo::DivOp::create(
          rewriter, loc, eigvalsType, ones, sqrtShifted);
      for (auto &ss : singleSolves) {
        rewriter.setInsertionPoint(ss.solve);
        auto resultType = cast<RankedTensorType>(ss.solve.getType());
        Value rhs = ss.solve.getB();

        if (LB) {
          rhs = stablehlo::TriangularSolveOp::create(
                    rewriter, loc, resultType, LB, rhs,
                    /*left_side=*/true, /*lower=*/true,
                    /*unit_diagonal=*/false, stablehlo::Transpose::NO_TRANSPOSE)
                    .getResult();
        }

        Value result = emitEigenSolve(rhs, invSqrtShifted, resultType);
        rewriter.replaceOp(ss.solve, result);
      }
    }

    // 7. Replace diagonal extractions
    //   Identity case:   diag(chol(K)) → sqrt(d)
    //   Generalized:     diag(chol(K)) → diag(L_B) .* sqrt(d)
    //   Both preserve log-det: sum(log(diag(L))) = 0.5 * log det(K).
    if (!diagExtractions.empty()) {
      Value sqrtShiftedForDiag =
          stablehlo::SqrtOp::create(rewriter, loc, eigvalsType, shifted);

      Value diagVec = sqrtShiftedForDiag;
      if (LB) {
        // Extract diag(L_B) via dot_general(L_B, I, batch=[1]x[1],
        // contract=[0]x[0])
        auto diagDotDims = stablehlo::DotDimensionNumbersAttr::get(
            rewriter.getContext(),
            /*lhsBatchingDims=*/{1}, /*rhsBatchingDims=*/{1},
            /*lhsContractingDims=*/{0}, /*rhsContractingDims=*/{0});
        SmallVector<Attribute> identityVals(
            N * N, rewriter.getFloatAttr(elemType, 0.0));
        for (int64_t i = 0; i < N; ++i)
          identityVals[i * N + i] = rewriter.getFloatAttr(elemType, 1.0);
        Value identity = stablehlo::ConstantOp::create(
            rewriter, loc, DenseElementsAttr::get(inputType, identityVals));
        Value diagLB = stablehlo::DotGeneralOp::create(
            rewriter, loc, eigvalsType, LB, identity, diagDotDims,
            /*precision_config=*/nullptr, /*algorithm=*/nullptr);
        // diag(chol(K)) ≈ diag(L_B) .* sqrt(d)
        diagVec = stablehlo::MulOp::create(rewriter, loc, eigvalsType, diagLB,
                                           sqrtShiftedForDiag);
      }

      for (auto *diagOp : diagExtractions) {
        rewriter.setInsertionPoint(diagOp);
        auto resultType =
            cast<RankedTensorType>(diagOp->getResult(0).getType());
        Value replacement;
        if (resultType.getRank() == 1) {
          replacement = diagVec;
        } else {
          Value bcast = stablehlo::BroadcastInDimOp::create(
              rewriter, loc, RankedTensorType::get({N, N}, elemType), diagVec,
              rewriter.getDenseI64ArrayAttr({0}));
          SmallVector<Attribute> identityVals(
              N * N, rewriter.getFloatAttr(elemType, 0.0));
          for (int64_t i = 0; i < N; ++i)
            identityVals[i * N + i] = rewriter.getFloatAttr(elemType, 1.0);
          Value identity = stablehlo::ConstantOp::create(
              rewriter, loc,
              DenseElementsAttr::get(RankedTensorType::get({N, N}, elemType),
                                     identityVals));
          replacement = stablehlo::MulOp::create(rewriter, loc, resultType,
                                                 bcast, identity);
        }
        rewriter.replaceOp(diagOp, replacement);
      }
    }

    // Erase Cholesky if no more users
    if (cholOp->use_empty())
      rewriter.eraseOp(cholOp);

    patternApplied = true;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Partial Inlining (Logpdf + Submodel)
//===----------------------------------------------------------------------===//

/// Compose two SymbolAttrs into a composite symbol by concatenating paths.
/// E.g., symbol<42> + symbol<17> -> symbol<42, 17>
static enzyme::SymbolAttr composeSymbols(enzyme::SymbolAttr outer,
                                         enzyme::SymbolAttr inner,
                                         MLIRContext *ctx) {
  SmallVector<uint64_t> composed(outer.getPath());
  composed.append(inner.getPath().begin(), inner.getPath().end());
  return enzyme::SymbolAttr::get(ctx, composed);
}

/// TODO: Move this to Enzyme proper
/// Flatten hierarchical addresses by composing the outer symbol with
/// the next symbol in each address that starts with outerSymbol.
/// E.g., [[<1>, <2>], [<3>]] with outerSymbol=<1> -> [[<1, 2>], [<3>]]
static ArrayAttr flattenAddressesForSymbol(ArrayAttr addresses,
                                           enzyme::SymbolAttr outerSymbol,
                                           MLIRContext *ctx) {
  SmallVector<Attribute> newAddresses;
  for (auto addr : addresses) {
    auto address = cast<ArrayAttr>(addr);
    if (address.size() >= 2 && address[0] == outerSymbol) {
      // Compose outer with next symbol, keep rest of address tail
      auto inner = cast<enzyme::SymbolAttr>(address[1]);
      auto composite = composeSymbols(outerSymbol, inner, ctx);
      SmallVector<Attribute> newAddr;
      newAddr.push_back(composite);
      for (unsigned i = 2; i < address.size(); ++i)
        newAddr.push_back(address[i]);
      newAddresses.push_back(ArrayAttr::get(ctx, newAddr));
    } else {
      newAddresses.push_back(addr);
    }
  }
  return ArrayAttr::get(ctx, newAddresses);
}

/// Inline submodel sample_region ops into the mcmc_region body.
///
/// A submodel sample_region has an empty logpdf region and no logpdf
/// attribute. Inlining dissolves the sample_region boundary, moving all
/// ops from the sampler body (including inner sample_region ops) into
/// the mcmc_region body.
///
/// Inner sample symbols are composed with the outer symbol to form unique
/// composite identifiers: symbol<outer> + symbol<inner> -> symbol<outer,
/// inner>. Addresses are flattened accordingly:
///   [[<outer>, <inner>]] -> [[<outer, inner>]]
static bool inlineSubmodelSampleRegions(MCMCRegionOp regionOp) {
  bool anyChanged = false;

  SmallVector<SampleRegionOp> sampleOps;
  regionOp.getSampler().walk(
      [&](SampleRegionOp op) { sampleOps.push_back(op); });

  for (SampleRegionOp sampleOp : sampleOps) {
    // Only handle submodel calls (empty logpdf region, no logpdf attribute)
    Region &logpdf = sampleOp.getLogpdf();
    if (!logpdf.empty())
      continue;
    if (sampleOp.getLogpdfFnAttr())
      continue;

    Region &sampler = sampleOp.getSampler();
    if (sampler.empty() || !sampler.hasOneBlock())
      continue;

    auto outerSymbol = sampleOp.getSymbolAttr();
    if (!outerSymbol)
      continue;

    Block &samplerEntry = sampler.front();
    auto *ctx = regionOp.getContext();

    // Map sampler block args to sample_region inputs
    OpBuilder builder(sampleOp);
    IRMapping mapper;
    auto inputs = sampleOp.getInputs();
    for (unsigned i = 0, e = samplerEntry.getNumArguments(); i < e; ++i) {
      if (i < inputs.size())
        mapper.map(samplerEntry.getArgument(i), inputs[i]);
    }

    // Clone all ops (except yield) into mcmc_region body.
    // Compose symbols on inner sample_region ops.
    for (Operation &op : samplerEntry.without_terminator()) {
      Operation *cloned = builder.clone(op, mapper);
      // Compose symbols on inner sample ops (both SampleRegionOp and SampleOp,
      // since --inline-mcmc-regions only converts top-level samples to regions)
      if (auto innerSample = dyn_cast<SampleRegionOp>(cloned)) {
        if (auto innerSymbol = innerSample.getSymbolAttr()) {
          innerSample.setSymbolAttr(
              composeSymbols(outerSymbol, innerSymbol, ctx));
        }
      } else if (auto innerSampleOp = dyn_cast<enzyme::SampleOp>(cloned)) {
        if (auto innerSymbol = innerSampleOp.getSymbolAttr()) {
          innerSampleOp.setSymbolAttr(
              composeSymbols(outerSymbol, innerSymbol, ctx));
        }
      }
    }

    // Replace sample_region results with mapped yield operands
    auto *yield = samplerEntry.getTerminator();
    for (auto [oldResult, yieldOperand] :
         llvm::zip(sampleOp.getResults(), yield->getOperands())) {
      oldResult.replaceAllUsesWith(mapper.lookupOrDefault(yieldOperand));
    }

    sampleOp.erase();

    // Flatten addresses: compose outer symbol into address entries
    if (auto allAddrs = regionOp.getAllAddressesAttr())
      regionOp.setAllAddressesAttr(
          flattenAddressesForSymbol(allAddrs, outerSymbol, ctx));
    if (auto sel = regionOp.getSelectionAttr())
      regionOp.setSelectionAttr(
          flattenAddressesForSymbol(sel, outerSymbol, ctx));

    anyChanged = true;
  }

  return anyChanged;
}

struct SICMPass : public enzyme::impl::SICMPassBase<SICMPass> {
  using SICMPassBase::SICMPassBase;

  /// Run SICM pattern rewrites + hoisting fixpoint on a specific region.
  bool runSICMOnRegion(MCMCRegionOp regionOp, AnalysisTarget target) {
    bool everChanged = false;
    for (int64_t iter = 0; iter < maxIterations; ++iter) {
      bool anyChanged = false;

      // Phase 1: Pattern rewrites with fresh analysis
      {
        SampleDependenceAnalysis analysis(regionOp, target);
        bool patternApplied = false;
        RewritePatternSet patterns(getOperation()->getContext());
        patterns.add<
            CholeskyScaleFactorizationHLO, CholeskyOuterProductScaleHLO,
            CholeskyEigenLiftHLO, DotGeneralScaleFactorizationHLO,
            TriangularSolveScaleFactorizationHLO, LogMultiplyDistributionHLO,
            // DotAbsorb patterns
            DotAbsorbDiagMulHLO, DotAbsorbFFTHLO, DotAbsorbTransposeHLO,
            DotAbsorbScatterHLO,
            // Element-wise scale factorization patterns
            PowerScaleFactorizationHLO, SqrtScaleFactorizationHLO,
            RsqrtScaleFactorizationHLO, AbsScaleFactorizationHLO,
            NegateDistributionHLO, ExpAddFactorizationHLO,
            DivideScaleFactorizationHLO>(analysis, patternApplied,
                                         getOperation()->getContext());
        GreedyRewriteConfig config;
        (void)applyPatternsGreedily(getOperation(), std::move(patterns),
                                    config);
        anyChanged |= patternApplied;
      }

      // Phase 2: Hoist (recomputes analysis internally)
      anyChanged |= hoistSampleInvariantOps(regionOp, target);

      everChanged |= anyChanged;
      if (!anyChanged)
        break;
    }
    return everChanged;
  }

  void runOnOperation() override {
    SmallVector<MCMCRegionOp> regions;
    getOperation()->walk([&](MCMCRegionOp op) { regions.push_back(op); });

    for (MCMCRegionOp regionOp : regions) {
      // Level 1: Optimize the sampler region.
      // Decompositions like chol(α²R) → α·chol(R) hoist chol(R) before
      // mcmc_region, making it available to the logpdf via inheritance.
      runSICMOnRegion(regionOp, AnalysisTarget::Sampler);

      // Construct unified logpdf region from per-site logpdf bodies.
      // Runs after sampler SICM so resolveValueForLogpdf inherits
      // optimized computation (e.g., hoisted chol(R) becomes a free
      // value captured by the logpdf).
      constructUnifiedLogpdf(regionOp);

      // Level 2: Optimize the logpdf region.
      // Catches logpdf-internal decompositions (e.g., MVN scale-family
      // where cholesky lives in the logpdf body, not the sampler).
      if (!regionOp.getLogpdf().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Logpdf);
    }
  }
};

} // end anonymous namespace
