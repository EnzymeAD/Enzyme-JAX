#include "Enzyme/MLIR/Analysis/SampleDependenceAnalysis.h"
#include "Enzyme/MLIR/Dialect/Impulse/Impulse.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Interfaces/ImpulseRegionUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/Support/Debug.h"

#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformPatterns.h.inc"

#define DEBUG_TYPE "sicm"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SICMPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

/// Build the sample-dependence analysis for the infer_region region (sampler
/// or logpdf) enclosing `op`; nullopt when `op` is outside an infer_region.
/// dialect.
static std::optional<SampleDependenceAnalysis>
analyzeEnclosingRegion(Operation *op) {
  auto regionOp = op->getParentOfType<InferRegionOp>();
  if (!regionOp)
    return std::nullopt;
  Region *region = op->getParentRegion();
  while (region && region->getParentOp() != regionOp)
    region = region->getParentOp()->getParentRegion();
  if (!region)
    return std::nullopt;
  AnalysisTarget target = region == &regionOp.getLogpdf()
                              ? AnalysisTarget::Logpdf
                              : AnalysisTarget::Sampler;
  return SampleDependenceAnalysis(regionOp, target);
}

static bool isEffectivelyScalar(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType && tensorType.getNumElements() == 1;
}

static bool isNonnegativeSupport(SupportAttr support) {
  if (!support)
    return false;
  switch (support.getKind()) {
  case SupportKind::POSITIVE:
  case SupportKind::UNIT_INTERVAL:
    return true;
  case SupportKind::GREATER_THAN:
  case SupportKind::INTERVAL: {
    FloatAttr lower = support.getLowerBound();
    return lower && !lower.getValue().isNegative();
  }
  default:
    return false;
  }
}

/// Leading logpdf arguments are the selected sites' non-RNG results, in
/// selection order.
static SupportAttr getLogpdfArgSupport(BlockArgument arg) {
  auto regionOp = dyn_cast<InferRegionOp>(arg.getOwner()->getParentOp());
  if (!regionOp || arg.getOwner()->getParent() != &regionOp.getLogpdf())
    return {};
  ArrayAttr selection = regionOp.getSelectionAttr();
  if (!selection)
    return {};
  DenseMap<Attribute, SampleRegionOp> siteByAddress;
  regionOp.getSampler().walk([&](SampleRegionOp site) {
    if (auto address = site.getAddressAttr())
      siteByAddress[address] = site;
  });
  unsigned argBegin = 0;
  for (Attribute addr : selection) {
    SampleRegionOp site = siteByAddress.lookup(addr);
    if (!site)
      return {};
    unsigned numOutputs = site.getNumResults() - 1;
    if (arg.getArgNumber() < argBegin + numOutputs)
      return site.getSupportAttr();
    argBegin += numOutputs;
  }
  return {};
}

static bool isSplatEvenInteger(Value value) {
  DenseFPElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat())
    return false;
  APFloat v = attr.getSplatValue<APFloat>();
  if (!v.isInteger())
    return false;
  bool losesInfo = false;
  v.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return std::fmod(v.convertToDouble(), 2.0) == 0.0;
}

static bool isNonnegative(Value value, PatternRewriter &rewriter) {
  enzyme::NonNegativeResultAnalysis analysis;

  Operation *parent = value.getParentBlock()->getParentOp();
  auto regionOp = dyn_cast<InferRegionOp>(parent);
  if (!regionOp)
    regionOp = parent->getParentOfType<InferRegionOp>();
  if (regionOp) {
    regionOp.getSampler().walk([&](SampleRegionOp site) {
      if (!isNonnegativeSupport(site.getSupportAttr()))
        return;
      for (Value result : llvm::drop_begin(site->getResults()))
        analysis.assume(result, true);
    });
    if (!regionOp.getLogpdf().empty())
      for (BlockArgument arg : regionOp.getLogpdf().getArguments())
        if (isNonnegativeSupport(getLogpdfArgSupport(arg)))
          analysis.assume(arg, true);
  }

  return analysis.guaranteed(value, rewriter);
}

struct ScaledInvariant {
  Value scale;
  Value invariant;
};

template <typename BinOpT>
static FailureOr<ScaledInvariant>
matchScalarBinaryInvariant(Value value, SampleDependenceAnalysis &analysis) {
  auto binOp = value.getDefiningOp<BinOpT>();
  if (!binOp)
    return failure();

  Value lhs = binOp.getLhs();
  Value rhs = binOp.getRhs();

  auto lhsBroadcast = lhs.getDefiningOp<stablehlo::BroadcastInDimOp>();
  auto rhsBroadcast = rhs.getDefiningOp<stablehlo::BroadcastInDimOp>();

  Value scale, invariant;
  if (lhsBroadcast && isEffectivelyScalar(lhsBroadcast.getOperand())) {
    scale = lhsBroadcast.getOperand();
    invariant = rhs;
  } else if (rhsBroadcast && isEffectivelyScalar(rhsBroadcast.getOperand())) {
    scale = rhsBroadcast.getOperand();
    invariant = lhs;
  } else if (isEffectivelyScalar(lhs) && !isEffectivelyScalar(rhs)) {
    scale = lhs;
    invariant = rhs;
  } else if (isEffectivelyScalar(rhs) && !isEffectivelyScalar(lhs)) {
    scale = rhs;
    invariant = lhs;
  } else {
    return failure();
  }

  if (analysis.isSampleDependent(invariant))
    return failure();
  if (!analysis.isSampleDependent(scale))
    return failure();

  return ScaledInvariant{scale, invariant};
}

static FailureOr<ScaledInvariant>
matchScaledInvariant(Value value, SampleDependenceAnalysis &analysis) {
  return matchScalarBinaryInvariant<stablehlo::MulOp>(value, analysis);
}

static FailureOr<ScaledInvariant>
matchAddedInvariant(Value value, SampleDependenceAnalysis &analysis) {
  return matchScalarBinaryInvariant<stablehlo::AddOp>(value, analysis);
}

static Value reshapeToScalar(OpBuilder &builder, Location loc, Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getRank() == 0)
    return value;
  return stablehlo::ReshapeOp::create(
      builder, loc, RankedTensorType::get({}, type.getElementType()), value);
}

static Value broadcastScalarToShape(OpBuilder &rewriter, Location loc,
                                    Value scalar, RankedTensorType resultType) {
  Value scalarVal = reshapeToScalar(rewriter, loc, scalar);
  if (cast<RankedTensorType>(scalarVal.getType()).getShape() !=
      resultType.getShape())
    return stablehlo::BroadcastInDimOp::create(
        rewriter, loc, resultType, scalarVal,
        rewriter.getDenseI64ArrayAttr({}));
  return scalarVal;
}

struct CholeskyScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::CholeskyOp> {
  CholeskyScaleFactorizationHLO(MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::CholeskyOp choleskyOp,
                     SampleDependenceAnalysis &analysis,
                     PatternRewriter &rewriter) {
    auto scaled = matchScaledInvariant(choleskyOp.getA(), analysis);
    if (failed(scaled))
      return failure();
    if (!isNonnegative(scaled->scale, rewriter))
      return failure();
    return scaled;
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::CholeskyOp choleskyOp,
                            const ScaledInvariant &match) {
    Location loc = choleskyOp.getLoc();
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto sqrtScale = stablehlo::SqrtOp::create(rewriter, loc, scalarScale);
    auto newCholesky = stablehlo::CholeskyOp::create(
        rewriter, loc, choleskyOp.getType(), match.invariant,
        choleskyOp.getLowerAttr());
    auto resultType = cast<RankedTensorType>(choleskyOp.getType());
    Value broadcastedSqrt =
        broadcastScalarToShape(rewriter, loc, sqrtScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedSqrt,
                                    newCholesky);
  }

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp choleskyOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(choleskyOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(choleskyOp, *analysis, rewriter);
    if (failed(match))
      return failure();
    rewriter.replaceOp(choleskyOp, applyRewrite(rewriter, choleskyOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: CholeskyScale fired\n");
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

struct CholeskyOuterProductScaleHLO
    : public OpRewritePattern<stablehlo::CholeskyOp> {
  CholeskyOuterProductScaleHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  struct Match {
    OuterProductInfo outerProd;
    Value invariantMatrix;
  };

  static FailureOr<Match> checkPreconditions(stablehlo::CholeskyOp cholOp,
                                             SampleDependenceAnalysis &analysis,
                                             PatternRewriter &rewriter) {
    auto mulOp = cholOp.getA().getDefiningOp<stablehlo::MulOp>();
    if (!mulOp)
      return failure();

    for (auto [maybeDiag, maybeInvariant] :
         {std::pair{mulOp.getLhs(), mulOp.getRhs()},
          std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
      if (analysis.isSampleDependent(maybeInvariant))
        continue;
      auto outerProd = matchOuterProduct(maybeDiag);
      if (!outerProd.sourceVec)
        continue;
      if (!analysis.isSampleDependent(outerProd.sourceVec))
        continue;
      if (!isNonnegative(outerProd.sourceVec, rewriter))
        continue;
      return Match{outerProd, maybeInvariant};
    }
    return failure();
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::CholeskyOp cholOp, const Match &match) {
    Location loc = cholOp.getLoc();
    auto inputType = cast<RankedTensorType>(cholOp.getType());
    auto newChol = stablehlo::CholeskyOp::create(
        rewriter, loc, inputType, match.invariantMatrix, cholOp.getLowerAttr());
    Value scaleBcast = cholOp.getLower() ? match.outerProd.rowBroadcast
                                         : match.outerProd.colBroadcast;
    return stablehlo::MulOp::create(rewriter, loc, inputType, scaleBcast,
                                    newChol);
  }

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp cholOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(cholOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(cholOp, *analysis, rewriter);
    if (failed(match))
      return failure();
    rewriter.replaceOp(cholOp, applyRewrite(rewriter, cholOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: CholeskyOuterProductScale fired\n");
    return success();
  }
};

struct DotGeneralScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  DotGeneralScaleFactorizationHLO(MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  struct Match {
    ScaledInvariant scaled;
    bool scaledOnLhs;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::DotGeneralOp dotOp,
                     SampleDependenceAnalysis &analysis) {
    auto lhsScaled = matchScaledInvariant(dotOp.getLhs(), analysis);
    if (succeeded(lhsScaled))
      return Match{*lhsScaled, true};
    auto rhsScaled = matchScaledInvariant(dotOp.getRhs(), analysis);
    if (succeeded(rhsScaled))
      return Match{*rhsScaled, false};
    return failure();
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::DotGeneralOp dotOp, const Match &match) {
    Location loc = dotOp.getLoc();
    auto resultType = cast<RankedTensorType>(dotOp.getType());
    Value newLhs = match.scaledOnLhs ? match.scaled.invariant : dotOp.getLhs();
    Value newRhs = match.scaledOnLhs ? dotOp.getRhs() : match.scaled.invariant;

    auto newDot = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, newRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());

    Value broadcastedScale =
        broadcastScalarToShape(rewriter, loc, match.scaled.scale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedScale,
                                    newDot);
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(dotOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(dotOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(dotOp, applyRewrite(rewriter, dotOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: DotGeneralScale fired\n");
    return success();
  }
};

///   dependence: L and b sample-invariant, s sample-dependent;
struct TriangularSolveScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::TriangularSolveOp> {
  TriangularSolveScaleFactorizationHLO(MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::TriangularSolveOp solveOp,
                     SampleDependenceAnalysis &analysis) {
    if (solveOp.getUnitDiagonal())
      return failure();
    auto scaled = matchScaledInvariant(solveOp.getA(), analysis);
    if (failed(scaled))
      return failure();
    if (analysis.isSampleDependent(solveOp.getB()))
      return failure();
    return scaled;
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::TriangularSolveOp solveOp,
                            const ScaledInvariant &match) {
    Location loc = solveOp.getLoc();
    auto resultType = cast<RankedTensorType>(solveOp.getType());
    auto newSolve = stablehlo::TriangularSolveOp::create(
        rewriter, loc, resultType, match.invariant, solveOp.getB(),
        solveOp.getLeftSideAttr(), solveOp.getLowerAttr(),
        solveOp.getUnitDiagonalAttr(), solveOp.getTransposeAAttr());

    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto scalarType = cast<RankedTensorType>(scalarScale.getType());
    Value one = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(
            scalarType,
            rewriter.getFloatAttr(scalarType.getElementType(), 1.0)));
    Value invScale =
        stablehlo::DivOp::create(rewriter, loc, scalarType, one, scalarScale);

    Value broadcastedInvScale =
        broadcastScalarToShape(rewriter, loc, invScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType,
                                    broadcastedInvScale, newSolve);
  }

  LogicalResult matchAndRewrite(stablehlo::TriangularSolveOp solveOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(solveOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(solveOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(solveOp, applyRewrite(rewriter, solveOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: TriangularSolveScale fired\n");
    return success();
  }
};

/// log(s * A) = log(s) + log(A)
///   structure/dimension: operand is s * A with s effectively scalar;
///   dependence: A sample-invariant, s sample-dependent;
struct LogMultiplyDistributionHLO : public OpRewritePattern<stablehlo::LogOp> {
  LogMultiplyDistributionHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::LogOp logOp, SampleDependenceAnalysis &analysis,
                     PatternRewriter &rewriter) {
    auto scaled = matchScaledInvariant(logOp.getOperand(), analysis);
    if (failed(scaled))
      return failure();
    if (!isNonnegative(scaled->scale, rewriter))
      return failure();
    return scaled;
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::LogOp logOp,
                            const ScaledInvariant &match) {
    Location loc = logOp.getLoc();
    auto resultType = cast<RankedTensorType>(logOp.getType());
    auto logInvariant =
        stablehlo::LogOp::create(rewriter, loc, resultType, match.invariant);
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto logScale = stablehlo::LogOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);
    Value broadcastedLogScale =
        broadcastScalarToShape(rewriter, loc, logScale, resultType);
    return stablehlo::AddOp::create(rewriter, loc, resultType,
                                    broadcastedLogScale, logInvariant);
  }

  LogicalResult matchAndRewrite(stablehlo::LogOp logOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(logOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(logOp, *analysis, rewriter);
    if (failed(match))
      return failure();
    rewriter.replaceOp(logOp, applyRewrite(rewriter, logOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: LogMultiply fired\n");
    return success();
  }
};

static std::pair<Value, stablehlo::ReshapeOp> lookThroughReshape(Value rhs) {
  if (auto reshapeOp = rhs.getDefiningOp<stablehlo::ReshapeOp>())
    return {reshapeOp.getOperand(), reshapeOp};
  return {rhs, nullptr};
}

struct DotAbsorbDiagMulHLO : public OpRewritePattern<stablehlo::DotGeneralOp> {
  DotAbsorbDiagMulHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  struct Match {
    Value scale;
    Value inner;
    SmallVector<int64_t> lhsBcastDims;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::DotGeneralOp dotOp,
                     SampleDependenceAnalysis &analysis) {
    if (analysis.isSampleDependent(dotOp.getLhs()))
      return failure();

    auto mulOp = dotOp.getRhs().getDefiningOp<stablehlo::MulOp>();
    if (!mulOp)
      return failure();

    Value scale, inner;
    stablehlo::BroadcastInDimOp broadcast;
    for (auto [candidate, other] :
         {std::pair{mulOp.getLhs(), mulOp.getRhs()},
          std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
      auto bcast = candidate.getDefiningOp<stablehlo::BroadcastInDimOp>();
      if (bcast && !analysis.isSampleDependent(bcast.getOperand())) {
        broadcast = bcast;
        scale = bcast.getOperand();
        inner = other;
        break;
      }
    }
    if (!scale || !analysis.isSampleDependent(inner))
      return failure();

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractDims = dimNumbers.getRhsContractingDimensions();
    DenseMap<int64_t, int64_t> rhsToLhs;
    for (size_t i = 0; i < rhsContractDims.size(); ++i)
      rhsToLhs[rhsContractDims[i]] = lhsContractDims[i];

    SmallVector<int64_t> lhsBcastDims;
    for (int64_t d : broadcast.getBroadcastDimensions()) {
      auto it = rhsToLhs.find(d);
      if (it == rhsToLhs.end())
        return failure();
      lhsBcastDims.push_back(it->second);
    }

    return Match{scale, inner, lhsBcastDims};
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::DotGeneralOp dotOp, const Match &match) {
    Location loc = dotOp.getLoc();
    auto lhsType = cast<RankedTensorType>(dotOp.getLhs().getType());
    Value lhsBroadcasted = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, lhsType, match.scale,
        rewriter.getDenseI64ArrayAttr(match.lhsBcastDims));
    Value newLhs = stablehlo::MulOp::create(rewriter, loc, lhsType,
                                            dotOp.getLhs(), lhsBroadcasted);
    auto resultType = cast<RankedTensorType>(dotOp.getType());
    return stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, match.inner,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(dotOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(dotOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(dotOp, applyRewrite(rewriter, dotOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: DotAbsorbDiagMul fired\n");
    return success();
  }
};

/// Only FFT and IFFT: their matrix F is symmetric, so transforming A's
/// contracting slice gives A F. RFFT and IRFFT change type and extent.
struct DotAbsorbFFTHLO : public OpRewritePattern<stablehlo::DotGeneralOp> {
  DotAbsorbFFTHLO(MLIRContext *context, PatternBenefit benefit = 4)
      : OpRewritePattern(context, benefit) {}

  struct Match {
    stablehlo::FftOp fftOp;
    stablehlo::ReshapeOp reshapeOp;
    int64_t contractDim;
    SmallVector<int64_t> fftDims;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::DotGeneralOp dotOp,
                     SampleDependenceAnalysis &analysis) {
    if (analysis.isSampleDependent(dotOp.getLhs()))
      return failure();

    auto [rhsInner, reshapeOp] = lookThroughReshape(dotOp.getRhs());
    auto fftOp = rhsInner.getDefiningOp<stablehlo::FftOp>();
    if (!fftOp)
      return failure();
    if (fftOp.getFftType() != stablehlo::FftType::FFT &&
        fftOp.getFftType() != stablehlo::FftType::IFFT)
      return failure();
    if (!analysis.isSampleDependent(fftOp.getOperand()))
      return failure();

    auto lhsType = cast<RankedTensorType>(dotOp.getLhs().getType());
    auto lhsContractDims =
        dotOp.getDotDimensionNumbers().getLhsContractingDimensions();
    if (lhsContractDims.size() != 1)
      return failure();
    int64_t contractDim = lhsContractDims[0];
    if (contractDim != 0 && contractDim != lhsType.getRank() - 1)
      return failure();

    SmallVector<int64_t> fftDims(fftOp.getFftLength().begin(),
                                 fftOp.getFftLength().end());
    int64_t fftProduct = 1;
    for (int64_t d : fftDims)
      fftProduct *= d;
    if (lhsType.getShape()[contractDim] != fftProduct)
      return failure();

    return Match{fftOp, reshapeOp, contractDim, fftDims};
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::DotGeneralOp dotOp, Match match) {
    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsShape = lhsType.getShape();
    int64_t lhsRank = lhsType.getRank();
    int64_t contractDim = match.contractDim;

    SmallVector<int64_t> expandedLhsShape;
    for (int64_t i = 0; i < contractDim; ++i)
      expandedLhsShape.push_back(lhsShape[i]);
    for (int64_t d : match.fftDims)
      expandedLhsShape.push_back(d);
    for (int64_t i = contractDim + 1; i < lhsRank; ++i)
      expandedLhsShape.push_back(lhsShape[i]);

    auto expandedLhsType =
        RankedTensorType::get(expandedLhsShape, lhsType.getElementType());
    Value expandedLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, expandedLhsType, lhs);

    // fft acts on the trailing dims, so move FFT dims at the front there.
    int64_t numFftDims = match.fftDims.size();
    int64_t numBatchAfter = lhsRank - contractDim - 1;
    Value toFFT = expandedLhs;

    if (contractDim == 0 && numBatchAfter > 0) {
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

    auto fftedLhs =
        stablehlo::FftOp::create(rewriter, loc, toFFT, match.fftOp.getFftType(),
                                 rewriter.getDenseI64ArrayAttr(match.fftDims));

    Value reshapeInput = fftedLhs;
    if (contractDim == 0 && numBatchAfter > 0) {
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

    Value newLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, lhsType, reshapeInput);

    Value newRhs;
    if (match.reshapeOp) {
      auto reshapeType = cast<RankedTensorType>(match.reshapeOp.getType());
      newRhs = stablehlo::ReshapeOp::create(rewriter, loc, reshapeType,
                                            match.fftOp.getOperand());
    } else {
      newRhs = match.fftOp.getOperand();
    }

    auto resultType = cast<RankedTensorType>(dotOp.getType());
    return stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, newRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(dotOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(dotOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(dotOp, applyRewrite(rewriter, dotOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: DotAbsorbFFT fired\n");
    return success();
  }
};

struct DotAbsorbTransposeHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  DotAbsorbTransposeHLO(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern(context, benefit) {}

  struct Match {
    stablehlo::TransposeOp transposeOp;
    stablehlo::ReshapeOp reshapeOp;
    int64_t contractDim;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::DotGeneralOp dotOp,
                     SampleDependenceAnalysis &analysis) {
    if (analysis.isSampleDependent(dotOp.getLhs()))
      return failure();

    auto [rhsInner, reshapeOp] = lookThroughReshape(dotOp.getRhs());
    auto transposeOp = rhsInner.getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();
    if (!analysis.isSampleDependent(transposeOp.getOperand()))
      return failure();

    auto lhsType = cast<RankedTensorType>(dotOp.getLhs().getType());
    auto lhsContractDims =
        dotOp.getDotDimensionNumbers().getLhsContractingDimensions();
    if (lhsContractDims.size() != 1)
      return failure();
    int64_t contractDim = lhsContractDims[0];
    if (contractDim != 0 && contractDim != lhsType.getRank() - 1)
      return failure();

    auto inputShape =
        cast<RankedTensorType>(transposeOp.getOperand().getType()).getShape();
    int64_t inputProduct = 1;
    for (int64_t d : inputShape)
      inputProduct *= d;
    if (lhsType.getShape()[contractDim] != inputProduct)
      return failure();

    return Match{transposeOp, reshapeOp, contractDim};
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::DotGeneralOp dotOp, Match match) {
    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsShape = lhsType.getShape();
    int64_t lhsRank = lhsType.getRank();
    int64_t contractDim = match.contractDim;

    Value transposeInput = match.transposeOp.getOperand();
    auto inputShape =
        cast<RankedTensorType>(transposeInput.getType()).getShape();
    auto perm = match.transposeOp.getPermutation();

    SmallVector<int64_t> expandedLhsShape;
    for (int64_t i = 0; i < contractDim; ++i)
      expandedLhsShape.push_back(lhsShape[i]);
    auto transposedShape =
        cast<RankedTensorType>(match.transposeOp.getType()).getShape();
    for (int64_t d : transposedShape)
      expandedLhsShape.push_back(d);
    for (int64_t i = contractDim + 1; i < lhsRank; ++i)
      expandedLhsShape.push_back(lhsShape[i]);

    auto expandedLhsType =
        RankedTensorType::get(expandedLhsShape, lhsType.getElementType());
    Value expandedLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, expandedLhsType, lhs);

    int64_t numBatchBefore = contractDim;
    int64_t numBatchAfter = lhsRank - contractDim - 1;
    SmallVector<int64_t> invPerm(numBatchBefore + perm.size() + numBatchAfter);
    for (int64_t i = 0; i < numBatchBefore; ++i)
      invPerm[i] = i;
    SmallVector<int64_t> innerInvPerm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
      innerInvPerm[perm[i]] = i;
    for (size_t i = 0; i < innerInvPerm.size(); ++i)
      invPerm[numBatchBefore + i] = numBatchBefore + innerInvPerm[i];
    for (int64_t i = 0; i < numBatchAfter; ++i)
      invPerm[numBatchBefore + perm.size() + i] =
          numBatchBefore + perm.size() + i;

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

    Value newLhs =
        stablehlo::ReshapeOp::create(rewriter, loc, lhsType, transposedLhs);

    Value newRhs;
    if (match.reshapeOp) {
      auto reshapeType = cast<RankedTensorType>(match.reshapeOp.getType());
      newRhs = stablehlo::ReshapeOp::create(rewriter, loc, reshapeType,
                                            transposeInput);
    } else {
      newRhs = transposeInput;
    }

    auto resultType = cast<RankedTensorType>(dotOp.getType());
    return stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, newLhs, newRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(dotOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(dotOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(dotOp, applyRewrite(rewriter, dotOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: DotAbsorbTranspose fired\n");
    return success();
  }
};

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
        return {nullptr, -1};
      channelDim = i;
    } else if (starts[i] == 0 && limits[i] == sourceShape[i] &&
               stridesArr[i] == 1) {
      // A full slice of a non-channel dim is allowed.
    } else {
      return {nullptr, -1};
    }
  }

  if (channelDim == -1)
    return {nullptr, -1};
  return {sliceOp.getOperand(), channelDim};
}

/// An overwrite scatter keeps one update per position but the gathered dot
/// sums all; without unique_indices keep only the last (StableHLO permits it).
struct DotAbsorbScatterHLO : public OpRewritePattern<stablehlo::DotGeneralOp> {
  DotAbsorbScatterHLO(MLIRContext *context, PatternBenefit benefit = 5)
      : OpRewritePattern(context, benefit) {}

  static std::pair<stablehlo::ScatterOp, Value>
  matchScatterIntoZeros(Value val) {
    auto scatterOp = val.getDefiningOp<stablehlo::ScatterOp>();
    if (!scatterOp || scatterOp.getInputs().size() != 1 ||
        scatterOp.getUpdates().size() != 1)
      return {nullptr, nullptr};

    Value input = scatterOp.getInputs()[0];
    DenseElementsAttr constAttr;
    if (!matchPattern(input, m_Constant(&constAttr)) || !constAttr.isSplat() ||
        !constAttr.getSplatValue<APFloat>().isZero())
      return {nullptr, nullptr};

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

  enum class DataKind { Direct, ComplexPair, ChannelComplex };

  struct Match {
    stablehlo::ScatterOp scatterOp;
    Value scatterIndices;
    stablehlo::ReshapeOp outerReshape;
    int64_t contractDim;
    SmallVector<int64_t> strideShape;
    SmallVector<int64_t> strideDimMap;
    DataKind kind;
    Value data;
    Value dataRe, dataIm;
    Value updates;
    int64_t channelUpdateDim;
    bool uniqueIndices = false;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::DotGeneralOp dotOp,
                     SampleDependenceAnalysis &analysis) {
    if (analysis.isSampleDependent(dotOp.getLhs()))
      return failure();

    auto [rhsInner, outerReshape] = lookThroughReshape(dotOp.getRhs());

    Match match;
    match.outerReshape = outerReshape;

    auto [directScatter, directData] = matchScatterIntoZeros(rhsInner);
    if (directScatter) {
      match.kind = DataKind::Direct;
      match.scatterOp = directScatter;
      match.scatterIndices = directScatter.getScatterIndices();
      match.data = directData;
      match.uniqueIndices = directScatter.getUniqueIndices();
    }

    if (!match.scatterOp) {
      if (auto complexOp = rhsInner.getDefiningOp<stablehlo::ComplexOp>()) {
        auto [scatterRe, dataRe] = matchScatterIntoZeros(complexOp.getLhs());
        auto [scatterIm, dataIm] = matchScatterIntoZeros(complexOp.getRhs());
        if (scatterRe && scatterIm &&
            scatterRe.getScatterIndices() == scatterIm.getScatterIndices()) {
          match.kind = DataKind::ComplexPair;
          match.scatterOp = scatterRe;
          match.scatterIndices = scatterRe.getScatterIndices();
          match.dataRe = dataRe;
          match.dataIm = dataIm;
          // The indices are shared, so either promise covers both.
          match.uniqueIndices =
              scatterRe.getUniqueIndices() || scatterIm.getUniqueIndices();
        }
      }
    }

    if (!match.scatterOp) {
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

      if (auto transposeOp =
              scatterInput.getDefiningOp<stablehlo::TransposeOp>()) {
        transposePerm.assign(transposeOp.getPermutation().begin(),
                             transposeOp.getPermutation().end());
        scatterInput = transposeOp.getOperand();
      }

      auto [scatter, updates] = matchScatterIntoZeros(scatterInput);
      if (!scatter)
        return failure();

      auto scatterDimNums = scatter.getScatterDimensionNumbers();
      auto updateWindowDims = scatterDimNums.getUpdateWindowDims();
      if (updateWindowDims.size() != 1)
        return failure();

      auto updatesType = cast<RankedTensorType>(updates.getType());
      auto updatesShape = updatesType.getShape();
      int64_t channelUpdateDim = updateWindowDims[0];
      if (updatesShape[channelUpdateDim] != 2 || updatesType.getRank() != 2)
        return failure();

      match.kind = DataKind::ChannelComplex;
      match.scatterOp = scatter;
      match.scatterIndices = scatter.getScatterIndices();
      match.updates = updates;
      match.channelUpdateDim = channelUpdateDim;
      match.uniqueIndices = scatter.getUniqueIndices();

      auto scatterInputShapeRef =
          cast<RankedTensorType>(scatter.getInputs()[0].getType()).getShape();
      auto scatterDimsToOp = scatterDimNums.getScatterDimsToOperandDims();

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

      for (size_t i = 0; i < transposedShapeVec.size(); ++i) {
        if ((int64_t)i != channelDimInTransposed)
          match.strideShape.push_back(transposedShapeVec[i]);
      }

      for (int64_t d = 0; d < (int64_t)scatterDimsToOp.size(); ++d) {
        int64_t operandDim = scatterDimsToOp[d];
        int64_t transposedDim = invPerm[operandDim];
        int64_t spatialDim = transposedDim > channelDimInTransposed
                                 ? transposedDim - 1
                                 : transposedDim;
        match.strideDimMap.push_back(spatialDim);
      }
    }

    if (analysis.isSampleDependent(match.scatterIndices))
      return failure();
    if (!analysis.isSampleDependent(match.scatterOp.getUpdates()[0]))
      return failure();

    if (match.strideShape.empty()) {
      auto scatterInputShape =
          cast<RankedTensorType>(match.scatterOp.getInputs()[0].getType())
              .getShape();
      match.strideShape.assign(scatterInputShape.begin(),
                               scatterInputShape.end());
      for (int64_t d : match.scatterOp.getScatterDimensionNumbers()
                           .getScatterDimsToOperandDims())
        match.strideDimMap.push_back(d);
    }

    int64_t scatterGridSize = 1;
    for (int64_t d : match.strideShape)
      scatterGridSize *= d;

    auto lhsType = cast<RankedTensorType>(dotOp.getLhs().getType());
    auto lhsContractDims =
        dotOp.getDotDimensionNumbers().getLhsContractingDimensions();
    if (lhsContractDims.size() != 1)
      return failure();
    match.contractDim = lhsContractDims[0];
    if (match.contractDim != 0 && match.contractDim != lhsType.getRank() - 1)
      return failure();
    if (lhsType.getShape()[match.contractDim] != scatterGridSize)
      return failure();

    return match;
  }

  static Value assembleData(PatternRewriter &rewriter, Location loc,
                            Match &match) {
    switch (match.kind) {
    case DataKind::Direct:
      return match.data;
    case DataKind::ComplexPair: {
      auto dataReType = cast<RankedTensorType>(match.dataRe.getType());
      auto complexElemType = ComplexType::get(dataReType.getElementType());
      auto complexDataType =
          RankedTensorType::get(dataReType.getShape(), complexElemType);
      return stablehlo::ComplexOp::create(rewriter, loc, complexDataType,
                                          match.dataRe, match.dataIm);
    }
    case DataKind::ChannelComplex: {
      auto updatesType = cast<RankedTensorType>(match.updates.getType());
      auto updatesShape = updatesType.getShape();
      auto elemType = updatesType.getElementType();
      int64_t channelUpdateDim = match.channelUpdateDim;
      int64_t dataK = updatesShape[1 - channelUpdateDim];

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
          rewriter, loc, slicedType, match.updates,
          rewriter.getDenseI64ArrayAttr(reStart),
          rewriter.getDenseI64ArrayAttr(reLimit),
          rewriter.getDenseI64ArrayAttr(unitStrides));
      Value imSlice = stablehlo::SliceOp::create(
          rewriter, loc, slicedType, match.updates,
          rewriter.getDenseI64ArrayAttr(imStart),
          rewriter.getDenseI64ArrayAttr(imLimit),
          rewriter.getDenseI64ArrayAttr(unitStrides));

      auto flatType = RankedTensorType::get({dataK}, elemType);
      Value reData =
          stablehlo::ReshapeOp::create(rewriter, loc, flatType, reSlice);
      Value imData =
          stablehlo::ReshapeOp::create(rewriter, loc, flatType, imSlice);

      auto complexElemType = ComplexType::get(elemType);
      auto complexDataType = RankedTensorType::get({dataK}, complexElemType);
      return stablehlo::ComplexOp::create(rewriter, loc, complexDataType,
                                          reData, imData);
    }
    }
    llvm_unreachable("unknown scatter data kind");
  }

  static Value maskToLastOccurrence(PatternRewriter &rewriter, Location loc,
                                    Value gathered, Value linearIdx, int64_t K,
                                    int64_t P, int64_t contractDim) {
    MLIRContext *ctx = rewriter.getContext();
    auto i64Type = rewriter.getI64Type();
    auto scalarI64 = RankedTensorType::get({}, i64Type);
    auto kType = RankedTensorType::get({K}, i64Type);
    auto pType = RankedTensorType::get({P}, i64Type);

    Value position = stablehlo::IotaOp::create(rewriter, loc, kType, 0);
    Value none = stablehlo::ConstantOp::create(
        rewriter, loc, pType, cast<ElementsAttr>(makeAttr(pType, -1)));
    auto winnerScatter = stablehlo::ScatterOp::create(
        rewriter, loc, ValueRange{none}, linearIdx, ValueRange{position},
        stablehlo::ScatterDimensionNumbersAttr::get(
            ctx, /*updateWindowDims=*/{}, /*insertedWindowDims=*/{0},
            /*inputBatchingDims=*/{}, /*scatterIndicesBatchingDims=*/{},
            /*scatterDimsToOperandDims=*/{0}, /*indexVectorDim=*/1),
        /*indicesAreSorted=*/false, /*uniqueIndices=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *body =
          rewriter.createBlock(&winnerScatter.getUpdateComputation(), {},
                               {scalarI64, scalarI64}, {loc, loc});
      rewriter.setInsertionPointToStart(body);
      Value later = stablehlo::MaxOp::create(
          rewriter, loc, body->getArgument(0), body->getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, later);
    }

    Value winnerAt = stablehlo::GatherOp::create(
        rewriter, loc, kType, winnerScatter.getResult(0), linearIdx,
        stablehlo::GatherDimensionNumbersAttr::get(
            ctx, /*offsetDims=*/{}, /*collapsedSliceDims=*/{0},
            /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
            /*startIndexMap=*/{0}, /*indexVectorDim=*/1),
        rewriter.getDenseI64ArrayAttr({1}));
    Value isLast = stablehlo::CompareOp::create(
        rewriter, loc, winnerAt, position, stablehlo::ComparisonDirection::EQ);

    // A 0/1 multiply, not a select, so while-loop LICM hoists it.
    auto gatheredType = cast<RankedTensorType>(gathered.getType());
    Value keep = stablehlo::ConvertOp::create(
        rewriter, loc,
        RankedTensorType::get({K}, gatheredType.getElementType()), isLast);
    Value keepB = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, gatheredType, keep,
        rewriter.getDenseI64ArrayAttr({contractDim}));
    return stablehlo::MulOp::create(rewriter, loc, gathered, keepB);
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::DotGeneralOp dotOp, Match match) {
    Location loc = dotOp.getLoc();
    Value lhs = dotOp.getLhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsShape = lhsType.getShape();
    int64_t contractDim = match.contractDim;

    Value newRhsData = assembleData(rewriter, loc, match);

    auto indicesType = cast<RankedTensorType>(match.scatterIndices.getType());
    int64_t K = indicesType.getShape()[0];

    SmallVector<int64_t> strides(match.strideShape.size());
    int64_t stride = 1;
    for (int64_t i = match.strideShape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= match.strideShape[i];
    }

    auto i64Type = rewriter.getI64Type();
    auto linearIdxType = RankedTensorType::get({K, 1}, i64Type);
    Value linearIdx = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(linearIdxType, rewriter.getI64IntegerAttr(0)));

    int64_t numIndexDims = match.strideDimMap.size();
    for (int64_t d = 0; d < numIndexDims; ++d) {
      auto sliceType =
          RankedTensorType::get({K, 1}, indicesType.getElementType());
      Value idxSlice = stablehlo::SliceOp::create(
          rewriter, loc, sliceType, match.scatterIndices,
          rewriter.getDenseI64ArrayAttr({0, d}),
          rewriter.getDenseI64ArrayAttr({K, d + 1}),
          rewriter.getDenseI64ArrayAttr({1, 1}));

      if (indicesType.getElementType() != i64Type) {
        idxSlice = stablehlo::ConvertOp::create(
            rewriter, loc, RankedTensorType::get({K, 1}, i64Type), idxSlice);
      }

      int64_t effectiveDim = match.strideDimMap[d];
      Value strideVal = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(linearIdxType, rewriter.getI64IntegerAttr(
                                                    strides[effectiveDim])));
      Value scaled = stablehlo::MulOp::create(rewriter, loc, linearIdxType,
                                              idxSlice, strideVal);
      linearIdx = stablehlo::AddOp::create(rewriter, loc, linearIdxType,
                                           linearIdx, scaled);
    }

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
    if (!match.uniqueIndices)
      gatheredLhs = maskToLastOccurrence(rewriter, loc, gatheredLhs, linearIdx,
                                         K, lhsShape[contractDim], contractDim);

    auto dataType = cast<RankedTensorType>(newRhsData.getType());
    Value finalRhs = newRhsData;
    if (match.outerReshape) {
      auto outerShape =
          cast<RankedTensorType>(match.outerReshape.getType()).getShape();
      SmallVector<int64_t> newRhsShape;
      newRhsShape.push_back(K);
      for (size_t i = 1; i < outerShape.size(); ++i)
        newRhsShape.push_back(outerShape[i]);
      auto newRhsType =
          RankedTensorType::get(newRhsShape, dataType.getElementType());
      finalRhs =
          stablehlo::ReshapeOp::create(rewriter, loc, newRhsType, newRhsData);
    }

    auto resultType = cast<RankedTensorType>(dotOp.getType());
    return stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, gatheredLhs, finalRhs,
        dotOp.getDotDimensionNumbersAttr(), dotOp.getPrecisionConfigAttr(),
        dotOp.getAlgorithmAttr());
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(dotOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(dotOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(dotOp, applyRewrite(rewriter, dotOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: DotAbsorbScatter fired\n");
    return success();
  }
};

struct PowerScaleFactorizationHLO : public OpRewritePattern<stablehlo::PowOp> {
  PowerScaleFactorizationHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::PowOp powOp, SampleDependenceAnalysis &analysis,
                     PatternRewriter &rewriter) {
    auto scaled = matchScaledInvariant(powOp.getLhs(), analysis);
    if (failed(scaled))
      return failure();
    if (!isNonnegative(scaled->scale, rewriter) &&
        !isSplatEvenInteger(powOp.getRhs()))
      return failure();
    return scaled;
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::PowOp powOp,
                            const ScaledInvariant &match) {
    Location loc = powOp.getLoc();
    Value exponent = powOp.getRhs();
    auto resultType = cast<RankedTensorType>(powOp.getType());
    auto powInvariant = stablehlo::PowOp::create(rewriter, loc, resultType,
                                                 match.invariant, exponent);
    Value broadcastedScale =
        broadcastScalarToShape(rewriter, loc, match.scale, resultType);
    auto powScale = stablehlo::PowOp::create(rewriter, loc, resultType,
                                             broadcastedScale, exponent);
    return stablehlo::MulOp::create(rewriter, loc, resultType, powScale,
                                    powInvariant);
  }

  LogicalResult matchAndRewrite(stablehlo::PowOp powOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(powOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(powOp, *analysis, rewriter);
    if (failed(match))
      return failure();
    rewriter.replaceOp(powOp, applyRewrite(rewriter, powOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: PowerScale fired\n");
    return success();
  }
};

struct SqrtScaleFactorizationHLO : public OpRewritePattern<stablehlo::SqrtOp> {
  SqrtScaleFactorizationHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::SqrtOp sqrtOp,
                     SampleDependenceAnalysis &analysis,
                     PatternRewriter &rewriter) {
    auto scaled = matchScaledInvariant(sqrtOp.getOperand(), analysis);
    if (failed(scaled))
      return failure();
    if (!isNonnegative(scaled->scale, rewriter))
      return failure();
    return scaled;
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::SqrtOp sqrtOp,
                            const ScaledInvariant &match) {
    Location loc = sqrtOp.getLoc();
    auto resultType = cast<RankedTensorType>(sqrtOp.getType());
    auto sqrtInvariant =
        stablehlo::SqrtOp::create(rewriter, loc, resultType, match.invariant);
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto sqrtScale = stablehlo::SqrtOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);
    Value broadcastedSqrt =
        broadcastScalarToShape(rewriter, loc, sqrtScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedSqrt,
                                    sqrtInvariant);
  }

  LogicalResult matchAndRewrite(stablehlo::SqrtOp sqrtOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sqrtOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(sqrtOp, *analysis, rewriter);
    if (failed(match))
      return failure();
    rewriter.replaceOp(sqrtOp, applyRewrite(rewriter, sqrtOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: SqrtScale fired\n");
    return success();
  }
};

struct RsqrtScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::RsqrtOp> {
  RsqrtScaleFactorizationHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::RsqrtOp rsqrtOp,
                     SampleDependenceAnalysis &analysis,
                     PatternRewriter &rewriter) {
    auto scaled = matchScaledInvariant(rsqrtOp.getOperand(), analysis);
    if (failed(scaled))
      return failure();
    if (!isNonnegative(scaled->scale, rewriter))
      return failure();
    return scaled;
  }

  static Value applyRewrite(PatternRewriter &rewriter,
                            stablehlo::RsqrtOp rsqrtOp,
                            const ScaledInvariant &match) {
    Location loc = rsqrtOp.getLoc();
    auto resultType = cast<RankedTensorType>(rsqrtOp.getType());
    auto rsqrtInvariant =
        stablehlo::RsqrtOp::create(rewriter, loc, resultType, match.invariant);
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto rsqrtScale = stablehlo::RsqrtOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);
    Value broadcastedRsqrt =
        broadcastScalarToShape(rewriter, loc, rsqrtScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedRsqrt,
                                    rsqrtInvariant);
  }

  LogicalResult matchAndRewrite(stablehlo::RsqrtOp rsqrtOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(rsqrtOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(rsqrtOp, *analysis, rewriter);
    if (failed(match))
      return failure();
    rewriter.replaceOp(rsqrtOp, applyRewrite(rewriter, rsqrtOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: RsqrtScale fired\n");
    return success();
  }
};

struct AbsScaleFactorizationHLO : public OpRewritePattern<stablehlo::AbsOp> {
  AbsScaleFactorizationHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::AbsOp absOp,
                     SampleDependenceAnalysis &analysis) {
    return matchScaledInvariant(absOp.getOperand(), analysis);
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::AbsOp absOp,
                            const ScaledInvariant &match) {
    Location loc = absOp.getLoc();
    auto resultType = cast<RankedTensorType>(absOp.getType());
    auto inputType = cast<RankedTensorType>(absOp.getOperand().getType());
    auto absInvariantType = RankedTensorType::get(inputType.getShape(),
                                                  resultType.getElementType());
    auto absInvariant = stablehlo::AbsOp::create(
        rewriter, loc, absInvariantType, match.invariant);
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto absScaleType = RankedTensorType::get({}, resultType.getElementType());
    auto absScale =
        stablehlo::AbsOp::create(rewriter, loc, absScaleType, scalarScale);
    Value broadcastedAbs =
        broadcastScalarToShape(rewriter, loc, absScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedAbs,
                                    absInvariant);
  }

  LogicalResult matchAndRewrite(stablehlo::AbsOp absOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(absOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(absOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(absOp, applyRewrite(rewriter, absOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: AbsScale fired\n");
    return success();
  }
};

struct NegateDistributionHLO : public OpRewritePattern<stablehlo::NegOp> {
  NegateDistributionHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::NegOp negOp,
                     SampleDependenceAnalysis &analysis) {
    return matchScaledInvariant(negOp.getOperand(), analysis);
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::NegOp negOp,
                            const ScaledInvariant &match) {
    Location loc = negOp.getLoc();
    auto resultType = cast<RankedTensorType>(negOp.getType());
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto negScale = stablehlo::NegOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);
    Value broadcastedNeg =
        broadcastScalarToShape(rewriter, loc, negScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedNeg,
                                    match.invariant);
  }

  LogicalResult matchAndRewrite(stablehlo::NegOp negOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(negOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(negOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(negOp, applyRewrite(rewriter, negOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: Negate fired\n");
    return success();
  }
};

struct ExpAddFactorizationHLO : public OpRewritePattern<stablehlo::ExpOp> {
  ExpAddFactorizationHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::ExpOp expOp,
                     SampleDependenceAnalysis &analysis) {
    return matchAddedInvariant(expOp.getOperand(), analysis);
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::ExpOp expOp,
                            const ScaledInvariant &match) {
    Location loc = expOp.getLoc();
    auto resultType = cast<RankedTensorType>(expOp.getType());
    auto expInvariant =
        stablehlo::ExpOp::create(rewriter, loc, resultType, match.invariant);
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scale);
    auto expScale = stablehlo::ExpOp::create(
        rewriter, loc, scalarScale.getType(), scalarScale);
    Value broadcastedExp =
        broadcastScalarToShape(rewriter, loc, expScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedExp,
                                    expInvariant);
  }

  LogicalResult matchAndRewrite(stablehlo::ExpOp expOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(expOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(expOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(expOp, applyRewrite(rewriter, expOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: ExpAdd fired\n");
    return success();
  }
};

struct DivideScaleFactorizationHLO : public OpRewritePattern<stablehlo::DivOp> {
  DivideScaleFactorizationHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  struct Match {
    ScaledInvariant scaled;
    bool scaledNumerator;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::DivOp divOp,
                     SampleDependenceAnalysis &analysis) {
    auto numScaled = matchScaledInvariant(divOp.getLhs(), analysis);
    if (succeeded(numScaled) && !analysis.isSampleDependent(divOp.getRhs()))
      return Match{*numScaled, true};
    auto denScaled = matchScaledInvariant(divOp.getRhs(), analysis);
    if (succeeded(denScaled))
      return Match{*denScaled, false};
    return failure();
  }

  static Value applyRewrite(PatternRewriter &rewriter, stablehlo::DivOp divOp,
                            const Match &match) {
    Location loc = divOp.getLoc();
    auto resultType = cast<RankedTensorType>(divOp.getType());

    if (match.scaledNumerator) {
      auto divInvariant = stablehlo::DivOp::create(
          rewriter, loc, resultType, match.scaled.invariant, divOp.getRhs());
      Value broadcastedScale =
          broadcastScalarToShape(rewriter, loc, match.scaled.scale, resultType);
      return stablehlo::MulOp::create(rewriter, loc, resultType,
                                      broadcastedScale, divInvariant);
    }

    auto newDiv = stablehlo::DivOp::create(
        rewriter, loc, resultType, divOp.getLhs(), match.scaled.invariant);
    Value scalarScale = reshapeToScalar(rewriter, loc, match.scaled.scale);
    auto scalarType = cast<RankedTensorType>(scalarScale.getType());
    Value one = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(
            scalarType,
            rewriter.getFloatAttr(scalarType.getElementType(), 1.0)));
    Value invScale =
        stablehlo::DivOp::create(rewriter, loc, scalarType, one, scalarScale);
    Value broadcastedInv =
        broadcastScalarToShape(rewriter, loc, invScale, resultType);
    return stablehlo::MulOp::create(rewriter, loc, resultType, broadcastedInv,
                                    newDiv);
  }

  LogicalResult matchAndRewrite(stablehlo::DivOp divOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(divOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(divOp, *analysis);
    if (failed(match))
      return failure();
    rewriter.replaceOp(divOp, applyRewrite(rewriter, divOp, *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: DivideScale fired\n");
    return success();
  }
};

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

static std::pair<Value, Value> decomposeScaledMatrix(Value value) {
  auto mulOp = value.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp)
    return {nullptr, nullptr};

  for (auto [maybeScalarBcast, maybeMatrix] :
       {std::pair{mulOp.getLhs(), mulOp.getRhs()},
        std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
    auto bcast = maybeScalarBcast.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (bcast && isEffectivelyScalar(bcast.getOperand()))
      return {bcast.getOperand(), maybeMatrix};
    if (isEffectivelyScalar(maybeScalarBcast))
      return {maybeScalarBcast, maybeMatrix};
  }
  return {nullptr, nullptr};
}

/// c * (s*A + t*B) with B = secondInvariant, diag(diagVector) or I; s may be
/// null and t may be folded into foldedDiagConst.
struct ScaledInvariantPlusDiag {
  Value invariant;
  Value scaleScalar;
  Value diagScalar;
  Value secondInvariant;
  FloatAttr foldedDiagConst;
  SmallVector<Value, 1> outerScales;
};

static ScaledInvariantPlusDiag
matchScaledInvariantPlusDiagSum(Value value,
                                SampleDependenceAnalysis &analysis) {
  auto addOp = value.getDefiningOp<stablehlo::AddOp>();
  if (!addOp)
    return {};

  for (auto [lhs, rhs] : {std::pair{addOp.getLhs(), addOp.getRhs()},
                          std::pair{addOp.getRhs(), addOp.getLhs()}}) {
    Value diagScalar = nullptr;
    FloatAttr foldedDiagConst = nullptr;
    Value secondInvariant = nullptr;
    auto [dScalar, dMatrix] = decomposeScaledMatrix(rhs);
    if (dScalar) {
      if (isIdentityMatrix(dMatrix)) {
        diagScalar = dScalar;
      } else if (!analysis.isSampleDependent(dMatrix)) {
        // Generalized case: rhs = t*B where B is invariant
        diagScalar = dScalar;
        secondInvariant = dMatrix;
      }
    }
    if (!diagScalar) {
      if (auto constOp = rhs.getDefiningOp<stablehlo::ConstantOp>()) {
        auto attr = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
        if (attr) {
          auto type = cast<RankedTensorType>(attr.getType());
          if (type.getRank() == 2 && type.getShape()[0] == type.getShape()[1]) {
            int64_t n = type.getShape()[0];
            auto elemType = type.getElementType();
            bool isDiagConst = true;
            APFloat diagVal =
                APFloat::getZero(cast<FloatType>(elemType).getFloatSemantics());
            bool first = true;
            for (auto [idx, val] : llvm::enumerate(attr.getValues<APFloat>())) {
              int64_t row = idx / n, col = idx % n;
              if (row == col) {
                if (first) {
                  diagVal = val;
                  first = false;
                } else if (val != diagVal) {
                  isDiagConst = false;
                  break;
                }
              } else if (!val.isZero()) {
                isDiagConst = false;
                break;
              }
            }
            if (isDiagConst && !first && !diagVal.isZero())
              foldedDiagConst = FloatAttr::get(elemType, diagVal);
          }
        }
      }
    }
    if (!diagScalar && !foldedDiagConst)
      continue;

    if (!analysis.isSampleDependent(lhs))
      return {lhs, nullptr, diagScalar, secondInvariant, foldedDiagConst};

    auto [sScalar, sMatrix] = decomposeScaledMatrix(lhs);
    if (sScalar && !analysis.isSampleDependent(sMatrix))
      return {sMatrix, sScalar, diagScalar, secondInvariant, foldedDiagConst};
  }
  return {};
}

static ScaledInvariantPlusDiag
matchScaledInvariantPlusDiagShift(Value value,
                                  SampleDependenceAnalysis &analysis) {
  SmallVector<Value, 1> outerScales;
  while (true) {
    ScaledInvariantPlusDiag match =
        matchScaledInvariantPlusDiagSum(value, analysis);
    if (match.invariant) {
      match.outerScales = std::move(outerScales);
      return match;
    }
    auto [scale, inner] = decomposeScaledMatrix(value);
    if (!scale)
      return {};
    outerScales.push_back(scale);
    value = inner;
  }
}

/// ADJOINT and TRANSPOSE coincide for the real factors these patterns target.
static bool isAdjointLikeTranspose(stablehlo::Transpose t) {
  return t == stablehlo::Transpose::ADJOINT ||
         t == stablehlo::Transpose::TRANSPOSE;
}

/// w = F^{-1} b (lower) or F^{-T} b (upper), so |w|^2 = b^T K^{-1} b. The solve
/// must read the triangle the Cholesky wrote; the other is unspecified.
static bool isWhiteningSolve(stablehlo::TriangularSolveOp solve,
                             bool cholLower) {
  if (!solve.getLeftSide() || solve.getUnitDiagonal() ||
      solve.getLower() != cholLower)
    return false;
  return cholLower ? solve.getTransposeA() == stablehlo::Transpose::NO_TRANSPOSE
                   : isAdjointLikeTranspose(solve.getTransposeA());
}

static bool isColoringSolve(stablehlo::TriangularSolveOp solve,
                            bool cholLower) {
  if (!solve.getLeftSide() || solve.getUnitDiagonal() ||
      solve.getLower() != cholLower)
    return false;
  return cholLower
             ? isAdjointLikeTranspose(solve.getTransposeA())
             : solve.getTransposeA() == stablehlo::Transpose::NO_TRANSPOSE;
}

static Value matchFullCholeskySolve(stablehlo::TriangularSolveOp outerSolve,
                                    Value cholResult, bool cholLower) {
  if (outerSolve.getA() != cholResult ||
      !isColoringSolve(outerSolve, cholLower))
    return nullptr;
  auto innerSolve =
      outerSolve.getB().getDefiningOp<stablehlo::TriangularSolveOp>();
  if (!innerSolve || innerSolve.getA() != cholResult ||
      !isWhiteningSolve(innerSolve, cholLower))
    return nullptr;
  return innerSolve.getB();
}

/// L .* I, dot_general(L, I) with one batching and one contracting dim in any
/// orientation (I is symmetric), or a stride-(n+1) slice of flat L or L^T.
static void collectDiagonalExtractions(Operation *user, Value cholResult,
                                       SmallVectorImpl<Operation *> &out) {
  auto cholType = cast<RankedTensorType>(cholResult.getType());
  if (cholType.getRank() != 2 ||
      cholType.getShape()[0] != cholType.getShape()[1])
    return;
  int64_t n = cholType.getShape()[0];

  if (auto mulOp = dyn_cast<stablehlo::MulOp>(user)) {
    for (auto [a, b] : {std::pair{mulOp.getLhs(), mulOp.getRhs()},
                        std::pair{mulOp.getRhs(), mulOp.getLhs()}}) {
      if (a == cholResult && isIdentityMatrix(b)) {
        out.push_back(user);
        return;
      }
    }
    return;
  }

  if (auto dotOp = dyn_cast<stablehlo::DotGeneralOp>(user)) {
    auto dims = dotOp.getDotDimensionNumbers();
    if (dims.getLhsBatchingDimensions().size() != 1 ||
        dims.getRhsBatchingDimensions().size() != 1 ||
        dims.getLhsContractingDimensions().size() != 1 ||
        dims.getRhsContractingDimensions().size() != 1)
      return;
    if (dims.getLhsBatchingDimensions()[0] ==
            dims.getLhsContractingDimensions()[0] ||
        dims.getRhsBatchingDimensions()[0] ==
            dims.getRhsContractingDimensions()[0])
      return;
    for (auto [a, b] : {std::pair{dotOp.getLhs(), dotOp.getRhs()},
                        std::pair{dotOp.getRhs(), dotOp.getLhs()}}) {
      if (a == cholResult && isIdentityMatrix(b)) {
        out.push_back(user);
        return;
      }
    }
    return;
  }

  if (auto transposeOp = dyn_cast<stablehlo::TransposeOp>(user)) {
    if (transposeOp.getPermutation() != ArrayRef<int64_t>{1, 0})
      return;
    for (Operation *tUser : transposeOp->getUsers())
      if (isa<stablehlo::ReshapeOp>(tUser))
        collectDiagonalExtractions(tUser, transposeOp.getResult(), out);
    return;
  }
  auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(user);
  if (!reshapeOp)
    return;
  auto flatType = cast<RankedTensorType>(reshapeOp.getType());
  if (flatType.getRank() != 1 || flatType.getShape()[0] != n * n)
    return;
  for (Operation *rUser : reshapeOp->getUsers()) {
    auto sliceOp = dyn_cast<stablehlo::SliceOp>(rUser);
    if (!sliceOp)
      continue;
    auto sliceType = cast<RankedTensorType>(sliceOp.getType());
    if (sliceType.getShape()[0] == n && sliceOp.getStartIndices()[0] == 0 &&
        sliceOp.getStrides()[0] == n + 1)
      out.push_back(sliceOp);
  }
}

static bool isSplatOne(Value value) {
  DenseFPElementsAttr attr;
  return matchPattern(value, m_Constant(&attr)) && attr.isSplat() &&
         attr.getSplatValue<APFloat>().isExactlyValue(1.0);
}

static bool isSingleInputReduce(stablehlo::ReduceOp reduce, OpOperand &use,
                                stablehlo::ReduceOpKind kind) {
  return reduce.getInputs().size() == 1 && use.getOperandNumber() == 0 &&
         stablehlo::CheckCommonReduceOp(reduce).kind == kind;
}

static bool isOnlyFullySummed(Value value) {
  int64_t rank = cast<RankedTensorType>(value.getType()).getRank();
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(user)) {
      if (!isOnlyFullySummed(reshape.getResult()))
        return false;
      continue;
    }
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(user)) {
      if (!isSingleInputReduce(reduce, use, stablehlo::ReduceOpKind::Add) ||
          (int64_t)reduce.getDimensions().size() != rank)
        return false;
      continue;
    }
    if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(user)) {
      auto dims = dot.getDotDimensionNumbers();
      bool onLhs = use.getOperandNumber() == 0;
      Value other = onLhs ? dot.getRhs() : dot.getLhs();
      auto contracting = onLhs ? dims.getLhsContractingDimensions()
                               : dims.getRhsContractingDimensions();
      if (!dims.getLhsBatchingDimensions().empty() ||
          (int64_t)contracting.size() != rank || !isSplatOne(other))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

/// The lift changes the diagonal entries but not their product, so only
/// sum(log(diag)) and prod(diag) uses are allowed.
static bool isLogDetOnlyVector(Value diag) {
  int64_t rank = cast<RankedTensorType>(diag.getType()).getRank();
  for (OpOperand &use : diag.getUses()) {
    Operation *user = use.getOwner();
    if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(user)) {
      if (!isLogDetOnlyVector(reshape.getResult()))
        return false;
      continue;
    }
    if (auto log = dyn_cast<stablehlo::LogOp>(user)) {
      if (!isOnlyFullySummed(log.getResult()))
        return false;
      continue;
    }
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(user)) {
      if (!isSingleInputReduce(reduce, use, stablehlo::ReduceOpKind::Mul) ||
          (int64_t)reduce.getDimensions().size() != rank)
        return false;
      continue;
    }
    return false;
  }
  return true;
}

/// F .* I (replaced by diag(sqrt d)) must first be reduced to the diagonal
/// vector: an add-reduce over one axis, or a dot_general contracting one axis
/// with a splat 1.0 vector.
static bool isLogDetOnlyDiagonal(Operation *extraction) {
  Value diag = extraction->getResult(0);
  if (cast<RankedTensorType>(diag.getType()).getRank() == 1)
    return isLogDetOnlyVector(diag);
  for (OpOperand &use : diag.getUses()) {
    Operation *user = use.getOwner();
    Value vector;
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(user)) {
      if (!isSingleInputReduce(reduce, use, stablehlo::ReduceOpKind::Add) ||
          reduce.getDimensions().size() != 1)
        return false;
      vector = reduce.getResult(0);
    } else if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(user)) {
      auto dims = dot.getDotDimensionNumbers();
      bool onLhs = use.getOperandNumber() == 0;
      Value other = onLhs ? dot.getRhs() : dot.getLhs();
      if (!dims.getLhsBatchingDimensions().empty() ||
          dims.getLhsContractingDimensions().size() != 1 ||
          cast<RankedTensorType>(other.getType()).getRank() != 1 ||
          !isSplatOne(other))
        return false;
      vector = dot.getResult();
    } else {
      return false;
    }
    if (!isLogDetOnlyVector(vector))
      return false;
  }
  return true;
}

/// factor) by W b with W = Q diag(1/sqrt d) Q^T (times L_B^{-1} when
class RotationConsumerCheck {
public:
  explicit RotationConsumerCheck(unsigned numSeeds) {
    for (unsigned i = 0; i < numSeeds; ++i)
      parent.push_back(i);
  }

  SmallVector<bool> run(ArrayRef<Value> seeds) {
    for (auto [i, seed] : llvm::enumerate(seeds))
      assign(seed, {/*product=*/false, /*axis=*/0, (unsigned)i});
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Form form = state.lookup(value);
      for (OpOperand &use : value.getUses())
        if (!visit(use, form))
          failed.push_back(form.group);
    }
    for (auto [value, group] : mustBeExact)
      if (state.count(value))
        failed.push_back(group);
    for (auto [value, group] : mustBeRotated)
      if (!state.count(value))
        failed.push_back(group);
    DenseSet<unsigned> failedRoots;
    for (unsigned group : failed)
      failedRoots.insert(find(group));
    SmallVector<bool> ok;
    for (unsigned i = 0; i < seeds.size(); ++i)
      ok.push_back(!failedRoots.contains(find(i)));
    return ok;
  }

private:
  struct Form {
    bool product;
    int64_t axis;
    unsigned group;
  };

  DenseMap<Value, Form> state;
  SmallVector<Value> worklist;
  SmallVector<unsigned> parent;
  SmallVector<unsigned> failed;
  SmallVector<std::pair<Value, unsigned>> mustBeExact;
  SmallVector<std::pair<Value, unsigned>> mustBeRotated;

  unsigned find(unsigned g) {
    while (parent[g] != g)
      g = parent[g] = parent[parent[g]];
    return g;
  }
  void unite(unsigned a, unsigned b) { parent[find(a)] = find(b); }

  void assign(Value value, Form form) {
    auto [it, inserted] = state.try_emplace(value, form);
    if (inserted) {
      worklist.push_back(value);
      return;
    }
    unite(it->second.group, form.group);
    if (it->second.product != form.product || it->second.axis != form.axis)
      failed.push_back(form.group);
  }

  static std::optional<int64_t>
  reshapeAxis(RankedTensorType in, RankedTensorType out, int64_t axis) {
    if (!in.hasStaticShape() || !out.hasStaticShape() ||
        in.getDimSize(axis) == 1)
      return std::nullopt;
    SmallVector<int64_t> inDims, outDims;
    for (int64_t i = 0; i < in.getRank(); ++i)
      if (in.getDimSize(i) != 1)
        inDims.push_back(i);
    for (int64_t i = 0; i < out.getRank(); ++i)
      if (out.getDimSize(i) != 1)
        outDims.push_back(i);
    if (inDims.size() != outDims.size())
      return std::nullopt;
    std::optional<int64_t> result;
    for (auto [i, o] : llvm::zip(inDims, outDims)) {
      if (in.getDimSize(i) != out.getDimSize(o))
        return std::nullopt;
      if (i == axis)
        result = o;
    }
    return result;
  }

  static bool isConstantAlong(Value value, int64_t axis) {
    DenseElementsAttr attr;
    if (matchPattern(value, m_Constant(&attr)))
      return attr.isSplat();
    auto bcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcast)
      return false;
    auto srcType = cast<RankedTensorType>(bcast.getOperand().getType());
    for (auto [i, d] : llvm::enumerate(bcast.getBroadcastDimensions()))
      if (d == axis && srcType.getDimSize(i) != 1)
        return false;
    return true;
  }

  bool visitBinary(Operation *op, unsigned idx, Form form) {
    Value other = op->getOperand(1 - idx);
    Value result = op->getResult(0);
    bool isMul = isa<stablehlo::MulOp>(op);
    bool isDiv = isa<stablehlo::DivOp>(op);
    auto it = state.find(other);
    if (it != state.end()) {
      Form otherForm = it->second;
      unite(form.group, otherForm.group);
      if (form.axis != otherForm.axis || isDiv)
        return false;
      if (isMul) {
        if (form.product || otherForm.product)
          return false;
        assign(result, {true, form.axis, form.group});
        return true;
      }
      if (form.product != otherForm.product)
        return false;
      assign(result, form);
      return true;
    }
    if (isMul || (isDiv && idx == 0)) {
      if (!isConstantAlong(other, form.axis))
        return false;
      mustBeExact.push_back({other, form.group});
      assign(result, form);
      return true;
    }
    if (isDiv)
      return false;
    // The other operand must become rotated; the op is handled from there.
    mustBeRotated.push_back({other, form.group});
    return true;
  }

  bool visitDot(stablehlo::DotGeneralOp dot, unsigned idx, Form form) {
    auto dims = dot.getDotDimensionNumbers();
    bool onLhs = idx == 0;
    Value other = onLhs ? dot.getRhs() : dot.getLhs();
    ArrayRef<int64_t> ownContract = onLhs ? dims.getLhsContractingDimensions()
                                          : dims.getRhsContractingDimensions();
    ArrayRef<int64_t> otherContract = onLhs
                                          ? dims.getRhsContractingDimensions()
                                          : dims.getLhsContractingDimensions();
    ArrayRef<int64_t> ownBatch = onLhs ? dims.getLhsBatchingDimensions()
                                       : dims.getRhsBatchingDimensions();
    if (llvm::is_contained(ownBatch, form.axis))
      return false;
    auto contractPos = llvm::find(ownContract, form.axis);
    bool contracted = contractPos != ownContract.end();

    if (form.product) {
      DenseElementsAttr attr;
      return contracted && matchPattern(other, m_Constant(&attr)) &&
             attr.isSplat();
    }

    auto it = state.find(other);
    if (it != state.end()) {
      // Exact iff both rotated axes are contracted together (O^T O = I).
      Form otherForm = it->second;
      unite(form.group, otherForm.group);
      if (!contracted || otherForm.product)
        return false;
      int64_t pos = contractPos - ownContract.begin();
      return otherContract[pos] == otherForm.axis;
    }
    if (contracted) {
      // Only valid as the rotated-rotated sink above.
      mustBeRotated.push_back({other, form.group});
      return true;
    }
    // r stays a free dimension: dot(O v, x) = O dot(v, x) for exact x.
    mustBeExact.push_back({other, form.group});
    int64_t axis = ownBatch.size();
    if (!onLhs)
      axis += cast<RankedTensorType>(dot.getLhs().getType()).getRank() -
              dims.getLhsBatchingDimensions().size() -
              dims.getLhsContractingDimensions().size();
    for (int64_t d = 0; d < form.axis; ++d)
      if (!llvm::is_contained(ownContract, d) &&
          !llvm::is_contained(ownBatch, d))
        ++axis;
    assign(dot.getResult(), {false, axis, form.group});
    return true;
  }

  bool visit(OpOperand &use, Form form) {
    Operation *user = use.getOwner();
    unsigned idx = use.getOperandNumber();
    if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(user)) {
      auto axis =
          reshapeAxis(cast<RankedTensorType>(reshape.getOperand().getType()),
                      cast<RankedTensorType>(reshape.getType()), form.axis);
      if (!axis)
        return false;
      assign(reshape.getResult(), {form.product, *axis, form.group});
      return true;
    }
    if (auto transpose = dyn_cast<stablehlo::TransposeOp>(user)) {
      auto perm = transpose.getPermutation();
      int64_t axis = llvm::find(perm, form.axis) - perm.begin();
      assign(transpose.getResult(), {form.product, axis, form.group});
      return true;
    }
    if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(user)) {
      auto resultType = cast<RankedTensorType>(bcast.getType());
      int64_t axis = bcast.getBroadcastDimensions()[form.axis];
      if (resultType.getDimSize(axis) !=
          cast<RankedTensorType>(bcast.getOperand().getType())
              .getDimSize(form.axis))
        return false;
      assign(bcast.getResult(), {form.product, axis, form.group});
      return true;
    }
    if (auto neg = dyn_cast<stablehlo::NegOp>(user)) {
      assign(neg.getResult(), form);
      return true;
    }
    if (isa<stablehlo::AddOp, stablehlo::SubtractOp, stablehlo::MulOp,
            stablehlo::DivOp>(user))
      return visitBinary(user, idx, form);
    if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(user))
      return visitDot(dot, idx, form);
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(user)) {
      if (!isSingleInputReduce(reduce, use, stablehlo::ReduceOpKind::Add))
        return false;
      auto dims = reduce.getDimensions();
      if (llvm::is_contained(dims, form.axis))
        return form.product;
      if (form.product)
        return false;
      int64_t axis = form.axis;
      for (int64_t d : dims)
        if (d < form.axis)
          --axis;
      assign(reduce.getResult(0), {false, axis, form.group});
      return true;
    }
    return false;
  }
};

/// factor are rewritten in terms of Q and d:
///   full solve chains F\(F'\b)  ->  Q diag(1/d) Q^T b
///   single solves F\b, F'\b    ->  Q diag(1/sqrt(d)) Q^T b
/// chol(B). Handles A + t*I (s = 1), a constant diagonal matrix folded by
/// recognized in every spelling of collectDiagonalExtractions.
/// Only the full solve chain is replaced exactly; the other two replacements
/// preserve a function of the user's result, not the result:
///   single solve:  Q diag(1/sqrt d) Q^T b = O (F^{-1} b) for an orthogonal O,
///                  so only norms |F^{-1} b|^2 (and inner products of two such
///                  solves) are preserved;
///     replaced diagonal only the log-determinant;
///     definiteness of the original cholesky operand.
struct CholeskyEigenLiftHLO : public OpRewritePattern<stablehlo::CholeskyOp> {
  CholeskyEigenLiftHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  struct FullSolveInfo {
    stablehlo::TriangularSolveOp outerSolve;
    stablehlo::TriangularSolveOp innerSolve;
    Value rhs;
  };
  struct SingleSolveInfo {
    stablehlo::TriangularSolveOp solve;
    bool isAdjoint;
  };
  struct Match {
    ScaledInvariantPlusDiag decomposed;
    SmallVector<FullSolveInfo> fullSolves;
    SmallVector<SingleSolveInfo> singleSolves;
    SmallVector<Operation *> diagExtractions;
  };

  static FailureOr<Match>
  checkPreconditions(stablehlo::CholeskyOp cholOp,
                     SampleDependenceAnalysis &analysis) {
    if (!analysis.isSampleDependent(cholOp.getA()))
      return failure();

    Match match;
    match.decomposed =
        matchScaledInvariantPlusDiagShift(cholOp.getA(), analysis);
    if (!match.decomposed.invariant)
      return failure();

    Value cholResult = cholOp.getResult();
    bool cholLower = cholOp.getLower();

    // Chains first, so their inner solves are not taken as single solves.
    DenseSet<Operation *> chainSolveOps;
    SmallVector<Operation *> diagCandidates;
    for (auto *user : cholResult.getUsers()) {
      auto triSolve = dyn_cast<stablehlo::TriangularSolveOp>(user);
      if (!triSolve) {
        collectDiagonalExtractions(user, cholResult, diagCandidates);
        continue;
      }

      if (!triSolve.getLeftSide())
        return failure();

      if (Value rhs = matchFullCholeskySolve(triSolve, cholResult, cholLower)) {
        auto innerSolve =
            triSolve.getB().getDefiningOp<stablehlo::TriangularSolveOp>();
        match.fullSolves.push_back({triSolve, innerSolve, rhs});
        chainSolveOps.insert(triSolve.getOperation());
        chainSolveOps.insert(innerSolve.getOperation());
      }
    }

    SmallVector<stablehlo::TriangularSolveOp> singleCandidates;
    for (auto *user : cholResult.getUsers()) {
      auto triSolve = dyn_cast<stablehlo::TriangularSolveOp>(user);
      if (!triSolve || chainSolveOps.contains(triSolve.getOperation()) ||
          !isWhiteningSolve(triSolve, cholLower))
        continue;
      if (llvm::is_contained(singleCandidates, triSolve))
        continue;
      singleCandidates.push_back(triSolve);
    }
    SmallVector<Value> seeds;
    for (auto solve : singleCandidates)
      seeds.push_back(solve.getResult());
    SmallVector<bool> normOnly = RotationConsumerCheck(seeds.size()).run(seeds);
    for (auto [solve, ok] : llvm::zip(singleCandidates, normOnly)) {
      if (ok)
        match.singleSolves.push_back({solve, !cholLower});
      else
        LLVM_DEBUG(llvm::dbgs() << "SICM: CholeskyEigenLift keeps a solve "
                                   "with a consumer that is not norm-only\n");
    }

    for (Operation *diagOp : diagCandidates) {
      if (isLogDetOnlyDiagonal(diagOp))
        match.diagExtractions.push_back(diagOp);
      else
        LLVM_DEBUG(llvm::dbgs() << "SICM: CholeskyEigenLift keeps a diagonal "
                                   "with a consumer that is not "
                                   "log-determinant-only\n");
    }

    if (match.fullSolves.empty() && match.singleSolves.empty() &&
        match.diagExtractions.empty())
      return failure();

    return match;
  }

  static void applyRewrite(PatternRewriter &rewriter,
                           stablehlo::CholeskyOp cholOp, Match &m) {
    Location loc = cholOp.getLoc();
    ScaledInvariantPlusDiag match = m.decomposed;
    auto &fullSolves = m.fullSolves;
    auto &singleSolves = m.singleSolves;
    auto &diagExtractions = m.diagExtractions;

    if (!match.diagScalar) {
      auto scalarType =
          RankedTensorType::get({}, match.foldedDiagConst.getType());
      match.diagScalar = stablehlo::ConstantOp::create(
          rewriter, loc, scalarType,
          DenseElementsAttr::get(scalarType, match.foldedDiagConst));
    }

    // Distribute outer scalar factors into the sum:
    // c * (s*A + t*B) = (c*s)*A + (c*t)*B, with s = 1 when absent.
    if (!match.outerScales.empty()) {
      Value c = reshapeToScalar(rewriter, loc, match.outerScales.front());
      for (Value factor : llvm::drop_begin(match.outerScales))
        c = stablehlo::MulOp::create(rewriter, loc, c,
                                     reshapeToScalar(rewriter, loc, factor));
      match.scaleScalar =
          match.scaleScalar
              ? stablehlo::MulOp::create(
                    rewriter, loc, c,
                    reshapeToScalar(rewriter, loc, match.scaleScalar))
                    .getResult()
              : c;
      match.diagScalar = stablehlo::MulOp::create(
          rewriter, loc, c, reshapeToScalar(rewriter, loc, match.diagScalar));
    }

    auto inputType = cast<RankedTensorType>(match.invariant.getType());
    int64_t N = inputType.getShape()[0];
    auto elemType = inputType.getElementType();
    auto eigvalsType = RankedTensorType::get({N}, elemType);
    auto infoType = RankedTensorType::get({}, rewriter.getIntegerType(64));

    // then eigendecompose C_w instead of A directly.
    Value LB;
    Value eigenTarget = match.invariant;
    if (match.secondInvariant) {
      LB = stablehlo::CholeskyOp::create(rewriter, loc, inputType,
                                         match.secondInvariant,
                                         rewriter.getBoolAttr(true))
               .getResult();

      Value Y = stablehlo::TriangularSolveOp::create(
                    rewriter, loc, inputType, LB, match.invariant,
                    /*left_side=*/true, /*lower=*/true,
                    /*unit_diagonal=*/false, stablehlo::Transpose::NO_TRANSPOSE)
                    .getResult();

      eigenTarget = stablehlo::TriangularSolveOp::create(
                        rewriter, loc, inputType, LB, Y,
                        /*left_side=*/false, /*lower=*/true,
                        /*unit_diagonal=*/false, stablehlo::Transpose::ADJOINT)
                        .getResult();
    }

    auto syevd = enzymexla::SyevdOp::create(
        rewriter, loc, TypeRange{inputType, eigvalsType, infoType}, eigenTarget,
        enzymexla::LapackUploAttr::get(rewriter.getContext(),
                                       enzymexla::LapackUplo::L));
    Value Q = syevd.getEigenvectors();
    Value lambda = syevd.getEigenvalues();

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

    if (!diagExtractions.empty()) {
      Value sqrtShiftedForDiag =
          stablehlo::SqrtOp::create(rewriter, loc, eigvalsType, shifted);

      Value diagVec = sqrtShiftedForDiag;
      if (LB) {
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

    if (cholOp->use_empty())
      rewriter.eraseOp(cholOp);
  }

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp cholOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(cholOp);
    if (!analysis)
      return failure();
    auto match = checkPreconditions(cholOp, *analysis);
    if (failed(match))
      return failure();
    applyRewrite(rewriter, cholOp, *match);
    LLVM_DEBUG(llvm::dbgs() << "SICM: CholeskyEigenLift fired\n");
    return success();
  }
};

// sample-invariant terms move to the y side. A non-constant scalar that is
enum class SicmAffineKind { Mul, Div, Add, SubRhs, SubLhs };
struct SicmAffineStep {
  SicmAffineKind kind;
  Value scalar;
};

/// ones column), negated when it enters the mean with a minus sign.
struct AffineMeanTerm {
  Value coef;
  Value col;
  bool negate;
  bool dependent;
};
struct AffineMeanOffset {
  Value v;
  bool negate;
};

/// Minimum removed work N * D (pass option `sufficient-stats-min-work`).
constexpr int64_t kGaussianStatsDefaultMinWork = 64;

struct GaussianStatsMatch {
  Value X, y, beta;
  int64_t N = 0, D = 0;
  int64_t xContractDim = 1;
  SmallVector<SicmAffineStep> affine; // outer-first
  bool hasFactor = false;
  Value factorScalar;
  bool factorIsDivide = false;
  bool affineMean = false;
  SmallVector<AffineMeanTerm> terms;
  SmallVector<AffineMeanOffset> offsets;
  bool squareSumRoot = false;
};

static bool isScalarShapedValue(Value v) {
  if (auto bc = v.getDefiningOp<stablehlo::BroadcastInDimOp>())
    return isEffectivelyScalar(bc.getOperand());
  DenseFPElementsAttr attr;
  if (matchPattern(v, m_Constant(&attr)) && attr.isSplat())
    return true;
  return isEffectivelyScalar(v);
}

static Value shapedToScalar(OpBuilder &b, Location loc, Value v,
                            Type elemType) {
  if (auto bc = v.getDefiningOp<stablehlo::BroadcastInDimOp>())
    return reshapeToScalar(b, loc, bc.getOperand());
  DenseFPElementsAttr attr;
  if (matchPattern(v, m_Constant(&attr)) && attr.isSplat()) {
    auto st = RankedTensorType::get({}, elemType);
    return stablehlo::ConstantOp::create(
        b, loc, st, DenseElementsAttr::get(st, attr.getSplatValue<APFloat>()));
  }
  return reshapeToScalar(b, loc, v);
}

static Value sicmScalarConst(OpBuilder &b, Location loc, Type elemType,
                             double v) {
  auto st = RankedTensorType::get({}, elemType);
  return stablehlo::ConstantOp::create(
      b, loc, st, DenseElementsAttr::get(st, b.getFloatAttr(elemType, v)));
}

static Value emitContractDot(OpBuilder &b, Location loc, RankedTensorType resTy,
                             Value lhs, Value rhs, ArrayRef<int64_t> lhsC,
                             ArrayRef<int64_t> rhsC) {
  auto dims = stablehlo::DotDimensionNumbersAttr::get(
      b.getContext(), /*lhsBatch=*/{}, /*rhsBatch=*/{}, lhsC, rhsC);
  return stablehlo::DotGeneralOp::create(b, loc, resTy, lhs, rhs, dims,
                                         /*precision_config=*/nullptr,
                                         /*algorithm=*/nullptr);
}

static bool isSplatFloat(Value v, double val) {
  DenseFPElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)) || !attr.isSplat())
    return false;
  APFloat sp = attr.getSplatValue<APFloat>();
  bool losesInfo = false;
  sp.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return sp.convertToDouble() == val;
}

struct SicmEventSum {
  Value K;
  int64_t N;
};

static bool isAddReduce(stablehlo::ReduceOp op) {
  Region &body = op.getBody();
  if (!body.hasOneBlock())
    return false;
  Block &blk = body.front();
  if (blk.getNumArguments() != 2)
    return false;
  auto ret = dyn_cast<stablehlo::ReturnOp>(blk.getTerminator());
  if (!ret || ret.getNumOperands() != 1)
    return false;
  auto add = ret.getOperand(0).getDefiningOp<stablehlo::AddOp>();
  if (!add)
    return false;
  Value a = blk.getArgument(0), b = blk.getArgument(1);
  return (add.getLhs() == a && add.getRhs() == b) ||
         (add.getLhs() == b && add.getRhs() == a);
}

static FailureOr<SicmEventSum> matchEventSumReduce(stablehlo::ReduceOp op) {
  if (op.getInputs().size() != 1 || op.getNumResults() != 1)
    return failure();
  auto resT = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!resT || resT.getRank() != 0)
    return failure();
  Value in = op.getInputs()[0];
  auto inT = dyn_cast<RankedTensorType>(in.getType());
  if (!inT || inT.getRank() != 1 || inT.isDynamicDim(0))
    return failure();
  auto dims = op.getDimensions();
  if (dims.size() != 1 || dims[0] != 0)
    return failure();
  if (!isAddReduce(op))
    return failure();
  if (!isSplatFloat(op.getInitValues()[0], 0.0))
    return failure();
  return SicmEventSum{in, inT.getDimSize(0)};
}

static FailureOr<SicmEventSum> matchEventSumDot(stablehlo::DotGeneralOp op) {
  auto resT = dyn_cast<RankedTensorType>(op.getType());
  if (!resT || resT.getRank() != 0)
    return failure();
  auto dn = op.getDotDimensionNumbersAttr();
  if (!dn.getLhsBatchingDimensions().empty() ||
      !dn.getRhsBatchingDimensions().empty())
    return failure();
  if (dn.getLhsContractingDimensions().size() != 1 ||
      dn.getRhsContractingDimensions().size() != 1)
    return failure();
  if (dn.getLhsContractingDimensions()[0] != 0 ||
      dn.getRhsContractingDimensions()[0] != 0)
    return failure();
  Value lhs = op.getLhs(), rhs = op.getRhs();
  auto lt = dyn_cast<RankedTensorType>(lhs.getType());
  auto rt = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lt || !rt || lt.getRank() != 1 || rt.getRank() != 1)
    return failure();
  if (isSplatFloat(rhs, 1.0) && !isSplatFloat(lhs, 1.0) && !lt.isDynamicDim(0))
    return SicmEventSum{lhs, lt.getDimSize(0)};
  if (isSplatFloat(lhs, 1.0) && !isSplatFloat(rhs, 1.0) && !rt.isDynamicDim(0))
    return SicmEventSum{rhs, rt.getDimSize(0)};
  return failure();
}

static Value matchSquareSumDot(stablehlo::DotGeneralOp op) {
  auto resT = dyn_cast<RankedTensorType>(op.getType());
  if (!resT || resT.getRank() != 0)
    return nullptr;
  if (op.getLhs() != op.getRhs())
    return nullptr;
  auto dn = op.getDotDimensionNumbersAttr();
  if (!dn.getLhsBatchingDimensions().empty() ||
      !dn.getRhsBatchingDimensions().empty())
    return nullptr;
  if (dn.getLhsContractingDimensions().size() != 1 ||
      dn.getRhsContractingDimensions().size() != 1 ||
      dn.getLhsContractingDimensions()[0] != 0 ||
      dn.getRhsContractingDimensions()[0] != 0)
    return nullptr;
  auto wt = dyn_cast<RankedTensorType>(op.getLhs().getType());
  if (!wt || wt.getRank() != 1 || wt.isDynamicDim(0))
    return nullptr;
  return op.getLhs();
}

static Value stripAffineScalarWrappers(Value K,
                                       SmallVectorImpl<SicmAffineStep> &steps) {
  Value cur = K;
  while (Operation *def = cur.getDefiningOp()) {
    if (auto mul = dyn_cast<stablehlo::MulOp>(def)) {
      Value a = mul.getLhs(), b = mul.getRhs();
      if (isScalarShapedValue(b) && !isScalarShapedValue(a)) {
        steps.push_back({SicmAffineKind::Mul, b});
        cur = a;
        continue;
      }
      if (isScalarShapedValue(a) && !isScalarShapedValue(b)) {
        steps.push_back({SicmAffineKind::Mul, a});
        cur = b;
        continue;
      }
      break;
    }
    if (auto add = dyn_cast<stablehlo::AddOp>(def)) {
      Value a = add.getLhs(), b = add.getRhs();
      if (isScalarShapedValue(b) && !isScalarShapedValue(a)) {
        steps.push_back({SicmAffineKind::Add, b});
        cur = a;
        continue;
      }
      if (isScalarShapedValue(a) && !isScalarShapedValue(b)) {
        steps.push_back({SicmAffineKind::Add, a});
        cur = b;
        continue;
      }
      break;
    }
    if (auto sub = dyn_cast<stablehlo::SubtractOp>(def)) {
      Value a = sub.getLhs(), b = sub.getRhs();
      if (isScalarShapedValue(b) && !isScalarShapedValue(a)) {
        steps.push_back({SicmAffineKind::SubRhs, b});
        cur = a;
        continue;
      }
      if (isScalarShapedValue(a) && !isScalarShapedValue(b)) {
        steps.push_back({SicmAffineKind::SubLhs, a});
        cur = b;
        continue;
      }
      break;
    }
    if (auto div = dyn_cast<stablehlo::DivOp>(def)) {
      Value a = div.getLhs(), b = div.getRhs();
      if (isScalarShapedValue(b) && !isScalarShapedValue(a)) {
        steps.push_back({SicmAffineKind::Div, b});
        cur = a;
        continue;
      }
      break;
    }
    break;
  }
  return cur;
}

static Value matchSquare(Value SQ) {
  if (auto mul = SQ.getDefiningOp<stablehlo::MulOp>())
    if (mul.getLhs() == mul.getRhs())
      return mul.getLhs();
  if (auto pw = SQ.getDefiningOp<stablehlo::PowOp>())
    if (isSplatFloat(pw.getRhs(), 2.0))
      return pw.getLhs();
  return nullptr;
}

static bool peelScalarFactor(Value W, Value &R, Value &factorScalar,
                             bool &isDivide) {
  if (auto div = W.getDefiningOp<stablehlo::DivOp>()) {
    if (isScalarShapedValue(div.getRhs()) &&
        !isScalarShapedValue(div.getLhs())) {
      R = div.getLhs();
      factorScalar = div.getRhs();
      isDivide = true;
      return true;
    }
  }
  if (auto mul = W.getDefiningOp<stablehlo::MulOp>()) {
    Value a = mul.getLhs(), b = mul.getRhs();
    if (isScalarShapedValue(b) && !isScalarShapedValue(a)) {
      R = a;
      factorScalar = b;
      isDivide = false;
      return true;
    }
    if (isScalarShapedValue(a) && !isScalarShapedValue(b)) {
      R = b;
      factorScalar = a;
      isDivide = false;
      return true;
    }
  }
  R = W;
  return false;
}

struct GaussianResidualMatch {
  Value X, y, beta;
  int64_t N, D;
  int64_t xContractDim;
  bool affineMean = false;
  SmallVector<AffineMeanTerm> terms;
  SmallVector<AffineMeanOffset> offsets;
};

/// Region-invariant non-constant scalars also stay coefficients, so that the
/// statistics depend on X and y only.
static Value getScalarCoefLeaf(Value v) {
  auto bc = v.getDefiningOp<stablehlo::BroadcastInDimOp>();
  if (!bc)
    return nullptr;
  Value s = bc.getOperand();
  if (!isEffectivelyScalar(s) || matchPattern(s, m_Constant()))
    return nullptr;
  return s;
}

/// the ones column (bare scalar), plus sample-invariant summands without such a
static bool collectAffineMeanTerms(Value v, bool negate, int64_t N,
                                   SampleDependenceAnalysis &analysis,
                                   SmallVectorImpl<AffineMeanTerm> &terms,
                                   SmallVectorImpl<AffineMeanOffset> &offsets,
                                   unsigned depth) {
  constexpr unsigned kMaxDepth = 16, kMaxTerms = 16;
  if (depth > kMaxDepth || terms.size() + offsets.size() >= kMaxTerms)
    return false;
  auto vt = dyn_cast<RankedTensorType>(v.getType());
  if (!vt || vt.getRank() != 1 || vt.isDynamicDim(0) || vt.getDimSize(0) != N)
    return false;
  if (Value s = getScalarCoefLeaf(v)) {
    terms.push_back({s, Value(), negate, analysis.isSampleDependent(s)});
    return true;
  }
  auto recurse = [&](Value w, bool neg) {
    return collectAffineMeanTerms(w, neg, N, analysis, terms, offsets,
                                  depth + 1);
  };
  bool invariant = !analysis.isSampleDependent(v);
  Operation *def = v.getDefiningOp();
  if (def) {
    if (auto add = dyn_cast<stablehlo::AddOp>(def))
      return recurse(add.getLhs(), negate) && recurse(add.getRhs(), negate);
    if (auto sub = dyn_cast<stablehlo::SubtractOp>(def))
      return recurse(sub.getLhs(), negate) && recurse(sub.getRhs(), !negate);
    if (auto neg = dyn_cast<stablehlo::NegOp>(def))
      return recurse(neg.getOperand(), !negate);
    if (auto mul = dyn_cast<stablehlo::MulOp>(def)) {
      for (auto [a, b] : {std::pair{mul.getLhs(), mul.getRhs()},
                          std::pair{mul.getRhs(), mul.getLhs()}}) {
        Value s = getScalarCoefLeaf(a);
        if (s && !analysis.isSampleDependent(b) && !getScalarCoefLeaf(b)) {
          terms.push_back({s, b, negate, analysis.isSampleDependent(s)});
          return true;
        }
      }
    }
  }
  if (invariant) {
    offsets.push_back({v, negate});
    return true;
  }
  return false;
}

/// Either subtract order: the residual is squared.
static FailureOr<GaussianResidualMatch>
matchGaussianResidual(Value R, SampleDependenceAnalysis &analysis) {
  auto sub = R.getDefiningOp<stablehlo::SubtractOp>();
  if (!sub)
    return failure();
  for (auto [maybeMean, maybeY] : {std::pair{sub.getLhs(), sub.getRhs()},
                                   std::pair{sub.getRhs(), sub.getLhs()}}) {
    auto dot = maybeMean.getDefiningOp<stablehlo::DotGeneralOp>();
    if (!dot)
      continue;
    if (analysis.isSampleDependent(maybeY))
      continue;
    auto yt = dyn_cast<RankedTensorType>(maybeY.getType());
    if (!yt || yt.getRank() != 1 || yt.isDynamicDim(0))
      continue;
    auto dn = dot.getDotDimensionNumbersAttr();
    if (!dn.getLhsBatchingDimensions().empty() ||
        !dn.getRhsBatchingDimensions().empty())
      continue;
    if (dn.getLhsContractingDimensions().size() != 1 ||
        dn.getRhsContractingDimensions().size() != 1)
      continue;
    Value l = dot.getLhs(), r = dot.getRhs();
    auto lt = dyn_cast<RankedTensorType>(l.getType());
    auto rt = dyn_cast<RankedTensorType>(r.getType());
    if (!lt || !rt)
      continue;

    Value X, beta;
    int64_t cd;
    if (lt.getRank() == 2 && rt.getRank() == 1 &&
        !analysis.isSampleDependent(l) && analysis.isSampleDependent(r)) {
      X = l;
      beta = r;
      cd = dn.getLhsContractingDimensions()[0];
    } else if (rt.getRank() == 2 && lt.getRank() == 1 &&
               !analysis.isSampleDependent(r) &&
               analysis.isSampleDependent(l)) {
      X = r;
      beta = l;
      cd = dn.getRhsContractingDimensions()[0];
    } else {
      continue;
    }
    auto xt = cast<RankedTensorType>(X.getType());
    int64_t nAxis = 1 - cd;
    if (xt.isDynamicDim(cd) || xt.isDynamicDim(nAxis))
      continue;
    int64_t N = xt.getDimSize(nAxis);
    int64_t D = xt.getDimSize(cd);
    if (N != yt.getDimSize(0))
      continue;
    return GaussianResidualMatch{X, maybeY, beta, N, D, cd};
  }
  for (auto [maybeMean, maybeY] : {std::pair{sub.getLhs(), sub.getRhs()},
                                   std::pair{sub.getRhs(), sub.getLhs()}}) {
    if (analysis.isSampleDependent(maybeY))
      continue;
    auto yt = dyn_cast<RankedTensorType>(maybeY.getType());
    if (!yt || yt.getRank() != 1 || yt.isDynamicDim(0))
      continue;
    GaussianResidualMatch rm;
    rm.N = yt.getDimSize(0);
    if (!collectAffineMeanTerms(maybeMean, /*negate=*/false, rm.N, analysis,
                                rm.terms, rm.offsets, /*depth=*/0) ||
        llvm::none_of(rm.terms,
                      [](const AffineMeanTerm &t) { return t.dependent; }))
      continue;
    rm.y = maybeY;
    // One column per distinct design column (terms sharing a column, e.g.
    // several bare scalars on the ones column, are merged).
    SmallVector<Value> distinctCols;
    for (const AffineMeanTerm &t : rm.terms)
      if (!llvm::is_contained(distinctCols, t.col))
        distinctCols.push_back(t.col);
    rm.D = distinctCols.size();
    rm.xContractDim = 1;
    rm.affineMean = true;
    return rm;
  }
  return failure();
}

static FailureOr<GaussianStatsMatch>
matchGaussianSquaredResidual(GaussianStatsMatch m, Value W, int64_t N,
                             SampleDependenceAnalysis &analysis, int64_t cThr,
                             int64_t minWork) {
  Value R;
  m.hasFactor = peelScalarFactor(W, R, m.factorScalar, m.factorIsDivide);
  auto rm = matchGaussianResidual(R, analysis);
  if (failed(rm))
    return failure();
  if (rm->N != N)
    return failure();
  if (rm->D <= 0 || rm->N < cThr * rm->D)
    return failure();
  if (rm->N * rm->D < minWork)
    return failure();
  m.X = rm->X;
  m.y = rm->y;
  m.beta = rm->beta;
  m.N = rm->N;
  m.D = rm->D;
  m.xContractDim = rm->xContractDim;
  m.affineMean = rm->affineMean;
  m.terms = rm->terms;
  m.offsets = rm->offsets;
  return m;
}

static FailureOr<GaussianStatsMatch>
matchGaussianSufficientStats(const SicmEventSum &es,
                             SampleDependenceAnalysis &analysis, int64_t cThr,
                             int64_t minWork) {
  GaussianStatsMatch m;
  Value SQ = stripAffineScalarWrappers(es.K, m.affine);
  Value W = matchSquare(SQ);
  if (!W)
    return failure();
  return matchGaussianSquaredResidual(m, W, es.N, analysis, cThr, minWork);
}

static FailureOr<GaussianStatsMatch>
matchGaussianSufficientStatsDot(stablehlo::DotGeneralOp sumOp,
                                SampleDependenceAnalysis &analysis,
                                int64_t cThr, int64_t minWork) {
  auto es = matchEventSumDot(sumOp);
  if (succeeded(es))
    return matchGaussianSufficientStats(*es, analysis, cThr, minWork);
  Value W = matchSquareSumDot(sumOp);
  if (!W)
    return failure();
  GaussianStatsMatch m;
  m.squareSumRoot = true;
  return matchGaussianSquaredResidual(
      m, W, cast<RankedTensorType>(W.getType()).getDimSize(0), analysis, cThr,
      minWork);
}

/// (invariant, hoistable) column; terms sharing a column are merged.
static void materializeAffineMean(OpBuilder &b, Location loc,
                                  GaussianStatsMatch &m) {
  if (!m.affineMean)
    return;
  Type elem = cast<RankedTensorType>(m.y.getType()).getElementType();
  auto colTy = RankedTensorType::get({m.N, 1}, elem);
  auto oneTy = RankedTensorType::get({1}, elem);
  SmallVector<Value> cols, coefs;
  Value onesCol;
  // A lone term negates its invariant, hoisted column rather than beta.
  SmallVector<Value> groupCols;
  SmallVector<SmallVector<const AffineMeanTerm *>> groups;
  for (const AffineMeanTerm &t : m.terms) {
    auto it = llvm::find(groupCols, t.col);
    if (it == groupCols.end()) {
      groupCols.push_back(t.col);
      groups.push_back({&t});
    } else {
      groups[it - groupCols.begin()].push_back(&t);
    }
  }
  for (auto &g : groups) {
    const AffineMeanTerm &t0 = *g.front();
    Value col;
    if (t0.col) {
      col = stablehlo::ReshapeOp::create(b, loc, colTy, t0.col);
    } else {
      if (!onesCol)
        onesCol = stablehlo::ConstantOp::create(
            b, loc, colTy,
            DenseElementsAttr::get(colTy, b.getFloatAttr(elem, 1.0)));
      col = onesCol;
    }
    auto asOne = [&](Value c) {
      return c.getType() == oneTy
                 ? c
                 : stablehlo::ReshapeOp::create(b, loc, oneTy, c).getResult();
    };
    Value c;
    if (g.size() == 1) {
      if (t0.negate)
        col = stablehlo::NegOp::create(b, loc, col);
      c = asOne(t0.coef);
    } else {
      for (const AffineMeanTerm *t : g) {
        Value ti = asOne(t->coef);
        if (!c)
          c = t->negate ? stablehlo::NegOp::create(b, loc, ti).getResult() : ti;
        else
          c = t->negate
                  ? stablehlo::SubtractOp::create(b, loc, c, ti).getResult()
                  : stablehlo::AddOp::create(b, loc, c, ti).getResult();
      }
    }
    cols.push_back(col);
    coefs.push_back(c);
  }
  m.X = cols.size() == 1
            ? cols.front()
            : stablehlo::ConcatenateOp::create(b, loc, cols, 1).getResult();
  m.beta = coefs.size() == 1
               ? coefs.front()
               : stablehlo::ConcatenateOp::create(b, loc, coefs, 0).getResult();
  Value y = m.y;
  for (const AffineMeanOffset &o : m.offsets)
    y = o.negate ? stablehlo::AddOp::create(b, loc, y, o.v).getResult()
                 : stablehlo::SubtractOp::create(b, loc, y, o.v).getResult();
  m.y = y;
  m.xContractDim = 1;
}

static Value getDesignND(OpBuilder &b, Location loc,
                         const GaussianStatsMatch &m) {
  if (m.xContractDim == 1)
    return m.X;
  auto xt = cast<RankedTensorType>(m.X.getType());
  auto ndTy = RankedTensorType::get({m.N, m.D}, xt.getElementType());
  return stablehlo::TransposeOp::create(b, loc, ndTy, m.X,
                                        b.getDenseI64ArrayAttr({1, 0}));
}

static std::pair<Value, Value> composeAffine(OpBuilder &b, Location loc,
                                             Type elemType,
                                             ArrayRef<SicmAffineStep> steps) {
  Value A = sicmScalarConst(b, loc, elemType, 1.0);
  Value B = sicmScalarConst(b, loc, elemType, 0.0);
  Value one = sicmScalarConst(b, loc, elemType, 1.0);
  Value zero = sicmScalarConst(b, loc, elemType, 0.0);
  for (const SicmAffineStep &st : llvm::reverse(steps)) {
    Value s = shapedToScalar(b, loc, st.scalar, elemType);
    Value alpha, beta;
    switch (st.kind) {
    case SicmAffineKind::Mul:
      alpha = s;
      beta = zero;
      break;
    case SicmAffineKind::Div:
      alpha = stablehlo::DivOp::create(b, loc, one, s);
      beta = zero;
      break;
    case SicmAffineKind::Add:
      alpha = one;
      beta = s;
      break;
    case SicmAffineKind::SubRhs: // current = inner - s
      alpha = one;
      beta = stablehlo::SubtractOp::create(b, loc, zero, s);
      break;
    case SicmAffineKind::SubLhs: // current = s - inner
      alpha = stablehlo::SubtractOp::create(b, loc, zero, one);
      beta = s;
      break;
    }
    Value newA = stablehlo::MulOp::create(b, loc, alpha, A);
    Value newB = stablehlo::AddOp::create(
        b, loc, stablehlo::MulOp::create(b, loc, alpha, B), beta);
    A = newA;
    B = newB;
  }
  return {A, B};
}

static Value finishGaussianScalar(OpBuilder &b, Location loc, Value q,
                                  const GaussianStatsMatch &m, Type elemType) {
  Value sumSQ = q;
  if (m.hasFactor) {
    Value s = shapedToScalar(b, loc, m.factorScalar, elemType);
    Value ssq = stablehlo::MulOp::create(b, loc, s, s);
    sumSQ = m.factorIsDivide
                ? stablehlo::DivOp::create(b, loc, q, ssq).getResult()
                : stablehlo::MulOp::create(b, loc, q, ssq).getResult();
  }
  if (m.squareSumRoot && m.affine.empty())
    return sumSQ;
  auto [A, B] = composeAffine(b, loc, elemType, m.affine);
  Value Aterm = stablehlo::MulOp::create(b, loc, A, sumSQ);
  Value Nc = sicmScalarConst(b, loc, elemType, (double)m.N);
  Value NB = stablehlo::MulOp::create(b, loc, Nc, B);
  return stablehlo::AddOp::create(b, loc, Aterm, NB);
}

static RankedTensorType qrInfoType(OpBuilder &b) {
  return RankedTensorType::get({}, b.getIntegerType(32));
}

/// Householder QR factors the leading D columns of [X y] as X alone, so
/// R = triu(P[:D, :D]), c = P[:D, D] and rem = P[D, D]^2.
static Value emitGaussianStatsQR(PatternRewriter &rewriter, Location loc,
                                 GaussianStatsMatch m) {
  materializeAffineMean(rewriter, loc, m);
  Type elem = cast<RankedTensorType>(m.X.getType()).getElementType();
  Value X = getDesignND(rewriter, loc, m);
  int64_t N = m.N, D = m.D;

  auto ddTy = RankedTensorType::get({D, D}, elem);
  auto dTy = RankedTensorType::get({D}, elem);
  auto scalarTy = RankedTensorType::get({}, elem);

  // The match guarantees N >= 2 D, so row D of P exists.
  Value yCol = stablehlo::ReshapeOp::create(
      rewriter, loc, RankedTensorType::get({N, 1}, elem), m.y);
  Value A =
      stablehlo::ConcatenateOp::create(rewriter, loc, ValueRange{X, yCol}, 1);
  auto augTy = RankedTensorType::get({N, D + 1}, elem);
  auto qr = enzymexla::QRFactorizationOp::create(
      rewriter, loc, TypeRange{augTy, augTy, qrInfoType(rewriter)}, A);
  Value P = qr.getR(); // R on and above the diagonal, reflectors below
  auto sliceP = [&](ArrayRef<int64_t> start, ArrayRef<int64_t> limit) -> Value {
    auto ty =
        RankedTensorType::get({limit[0] - start[0], limit[1] - start[1]}, elem);
    return stablehlo::SliceOp::create(rewriter, loc, ty, P,
                                      rewriter.getDenseI64ArrayAttr(start),
                                      rewriter.getDenseI64ArrayAttr(limit),
                                      rewriter.getDenseI64ArrayAttr({1, 1}));
  };
  // A 0/1 multiply, not a select, so while-loop LICM hoists it.
  Value RPacked = sliceP({0, 0}, {D, D});
  auto idxTy = RankedTensorType::get({D, D}, rewriter.getI64Type());
  Value rows = stablehlo::IotaOp::create(rewriter, loc, idxTy, 0);
  Value cols = stablehlo::IotaOp::create(rewriter, loc, idxTy, 1);
  Value upper = stablehlo::CompareOp::create(
      rewriter, loc, rows, cols, stablehlo::ComparisonDirection::LE);
  Value upperMask = stablehlo::ConvertOp::create(rewriter, loc, ddTy, upper);
  Value R = stablehlo::MulOp::create(rewriter, loc, RPacked, upperMask);
  Value c = stablehlo::ReshapeOp::create(rewriter, loc, dTy,
                                         sliceP({0, D}, {D, D + 1}));
  Value pDD = reshapeToScalar(rewriter, loc, sliceP({D, D}, {D + 1, D + 1}));
  Value rem = stablehlo::MulOp::create(rewriter, loc, pDD, pDD);

  Value Rb = emitContractDot(rewriter, loc, dTy, R, m.beta, {1}, {0});
  Value d = stablehlo::SubtractOp::create(rewriter, loc, c, Rb);
  Value dd = emitContractDot(rewriter, loc, scalarTy, d, d, {0}, {0});
  Value q = stablehlo::AddOp::create(rewriter, loc, dd, rem);
  return finishGaussianScalar(rewriter, loc, q, m, elem);
}

struct GaussianSufficientStatsHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  int64_t minWork;
  GaussianSufficientStatsHLO(MLIRContext *context, PatternBenefit benefit = 1,
                             int64_t minWork = kGaussianStatsDefaultMinWork)
      : OpRewritePattern(context, benefit), minWork(minWork) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto match =
        matchGaussianSufficientStatsDot(sumOp, *analysis, /*cThr=*/2, minWork);
    if (failed(match))
      return failure();
    rewriter.replaceOp(sumOp,
                       emitGaussianStatsQR(rewriter, sumOp.getLoc(), *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: GaussianSufficientStats (QR) fired\n");
    return success();
  }
};

struct GaussianSufficientStatsReduceHLO
    : public OpRewritePattern<stablehlo::ReduceOp> {
  int64_t minWork;
  GaussianSufficientStatsReduceHLO(
      MLIRContext *context, PatternBenefit benefit = 1,
      int64_t minWork = kGaussianStatsDefaultMinWork)
      : OpRewritePattern(context, benefit), minWork(minWork) {}

  LogicalResult matchAndRewrite(stablehlo::ReduceOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto es = matchEventSumReduce(sumOp);
    if (failed(es))
      return failure();
    auto match =
        matchGaussianSufficientStats(*es, *analysis, /*cThr=*/2, minWork);
    if (failed(match))
      return failure();
    rewriter.replaceOp(sumOp,
                       emitGaussianStatsQR(rewriter, sumOp.getLoc(), *match));
    LLVM_DEBUG(llvm::dbgs()
               << "SICM: GaussianSufficientStats (QR, reduce) fired\n");
    return success();
  }
};

static Value emitGaussianStatsGram(PatternRewriter &rewriter, Location loc,
                                   GaussianStatsMatch m) {
  materializeAffineMean(rewriter, loc, m);
  Type elem = cast<RankedTensorType>(m.X.getType()).getElementType();
  Value X = getDesignND(rewriter, loc, m);

  auto ddTy = RankedTensorType::get({m.D, m.D}, elem);
  auto dTy = RankedTensorType::get({m.D}, elem);
  auto scalarTy = RankedTensorType::get({}, elem);

  Value G = emitContractDot(rewriter, loc, ddTy, X, X, {0}, {0});
  Value Xty = emitContractDot(rewriter, loc, dTy, X, m.y, {0}, {0});
  Value yty = emitContractDot(rewriter, loc, scalarTy, m.y, m.y, {0}, {0});

  Value Gb = emitContractDot(rewriter, loc, dTy, G, m.beta, {1}, {0});
  Value bGb = emitContractDot(rewriter, loc, scalarTy, m.beta, Gb, {0}, {0});
  Value bXty = emitContractDot(rewriter, loc, scalarTy, m.beta, Xty, {0}, {0});
  Value two = sicmScalarConst(rewriter, loc, elem, 2.0);
  Value twobXty = stablehlo::MulOp::create(rewriter, loc, two, bXty);
  Value diff = stablehlo::SubtractOp::create(rewriter, loc, yty, twobXty);
  Value q = stablehlo::AddOp::create(rewriter, loc, diff, bGb);
  return finishGaussianScalar(rewriter, loc, q, m, elem);
}

/// The normal equations cancel badly at high R^2, worse in f32.
static bool gramElemTypeOk(const GaussianStatsMatch &m) {
  return cast<RankedTensorType>(m.y.getType()).getElementType().isF64();
}

struct GaussianSufficientStatsGramHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  int64_t minWork;
  GaussianSufficientStatsGramHLO(MLIRContext *context,
                                 PatternBenefit benefit = 1,
                                 int64_t minWork = kGaussianStatsDefaultMinWork)
      : OpRewritePattern(context, benefit), minWork(minWork) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto match =
        matchGaussianSufficientStatsDot(sumOp, *analysis, /*cThr=*/2, minWork);
    if (failed(match) || !gramElemTypeOk(*match))
      return failure();
    rewriter.replaceOp(sumOp,
                       emitGaussianStatsGram(rewriter, sumOp.getLoc(), *match));
    LLVM_DEBUG(llvm::dbgs() << "SICM: GaussianSufficientStats (Gram) fired\n");
    return success();
  }
};

struct GaussianSufficientStatsGramReduceHLO
    : public OpRewritePattern<stablehlo::ReduceOp> {
  int64_t minWork;
  GaussianSufficientStatsGramReduceHLO(
      MLIRContext *context, PatternBenefit benefit = 1,
      int64_t minWork = kGaussianStatsDefaultMinWork)
      : OpRewritePattern(context, benefit), minWork(minWork) {}

  LogicalResult matchAndRewrite(stablehlo::ReduceOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto es = matchEventSumReduce(sumOp);
    if (failed(es))
      return failure();
    auto match =
        matchGaussianSufficientStats(*es, *analysis, /*cThr=*/2, minWork);
    if (failed(match) || !gramElemTypeOk(*match))
      return failure();
    rewriter.replaceOp(sumOp,
                       emitGaussianStatsGram(rewriter, sumOp.getLoc(), *match));
    LLVM_DEBUG(llvm::dbgs()
               << "SICM: GaussianSufficientStats (Gram, reduce) fired\n");
    return success();
  }
};

struct SICMPass : public enzyme::impl::SICMPassBase<SICMPass> {
  using SICMPassBase::SICMPassBase;

  bool runSICMOnRegion(InferRegionOp regionOp, AnalysisTarget target) {
    bool everChanged = false;
    for (int64_t iter = 0; iter < maxIterations; ++iter) {
      bool anyChanged = false;

      RewritePatternSet patterns(getOperation()->getContext());
      mlir::transform::populateSicmPatterns(
          patterns, *getOperation()->getContext(), PatternBenefit(1));
      // default profitability threshold; re-add them with the pass option.
      if (sufficientStatsMinWork != kGaussianStatsDefaultMinWork) {
        auto &native = patterns.getNativePatterns();
        native.erase(
            llvm::remove_if(native,
                            [](const std::unique_ptr<RewritePattern> &p) {
                              return p->getDebugName().contains(
                                  "GaussianSufficientStats");
                            }),
            native.end());
        patterns
            .add<GaussianSufficientStatsHLO, GaussianSufficientStatsReduceHLO>(
                getOperation()->getContext(), PatternBenefit(1),
                sufficientStatsMinWork);
      }
      // Benefit 2 so the Gram form pre-empts the QR form on the same match.
      if (gramFormSufficientStats)
        patterns.add<GaussianSufficientStatsGramHLO,
                     GaussianSufficientStatsGramReduceHLO>(
            getOperation()->getContext(), PatternBenefit(2),
            sufficientStatsMinWork);
      GreedyRewriteConfig config;
      bool patternChanged = false;
      (void)applyPatternsGreedily(getOperation(), std::move(patterns), config,
                                  &patternChanged);
      anyChanged |= patternChanged;

      anyChanged |= hoistSampleInvariantOps(regionOp, target);

      everChanged |= anyChanged;
      if (!anyChanged)
        break;
    }
    return everChanged;
  }

  void runOnOperation() override {
    SmallVector<InferRegionOp> regions;
    getOperation()->walk([&](InferRegionOp op) { regions.push_back(op); });

    for (InferRegionOp regionOp : regions) {
      // Sampler first: what it hoists is inherited by the logpdf region built
      // next. The analysis needs a sampler entry block (none in custom logpdf).
      if (!regionOp.getSampler().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Sampler);

      constructLogpdfRegion(regionOp);

      if (!regionOp.getLogpdf().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Logpdf);
    }
  }
};

} // end anonymous namespace

// The tblgen'd populate* functions need the pattern classes above.
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformPatterns.cpp.inc"
