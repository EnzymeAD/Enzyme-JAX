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

///     scalar;
///   dependence: L and b sample-invariant, s sample-dependent;
struct TriangularSolveScaleFactorizationHLO
    : public OpRewritePattern<stablehlo::TriangularSolveOp> {
  TriangularSolveScaleFactorizationHLO(MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  static FailureOr<ScaledInvariant>
  checkPreconditions(stablehlo::TriangularSolveOp solveOp,
                     SampleDependenceAnalysis &analysis) {
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

///   structure: rhs (through an optional reshape) is fft(x); the single lhs
///     contracting dimension is the first or last and its extent equals the
///     product of the FFT lengths;
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

/// Detect chained triangular_solve pairs that compute Σ^{-1}b.
///
/// Pattern: triangular_solve(F, triangular_solve(F, b, ADJOINT), NO_TRANSPOSE)
/// where F is a Cholesky factor (upper or lower). The outer solve's `b` operand
/// IS the inner solve result. (The inner solve may use TRANSPOSE instead of
/// ADJOINT — equivalent for the real factors here.)
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
  // same Cholesky, with ADJOINT (or, equivalently, TRANSPOSE) transpose
  auto innerSolve =
      outerSolve.getB().getDefiningOp<stablehlo::TriangularSolveOp>();
  if (!innerSolve)
    return nullptr;
  if (innerSolve.getA() != cholResult)
    return nullptr;
  if (!innerSolve.getLeftSide())
    return nullptr;
  if (!isAdjointLikeTranspose(innerSolve.getTransposeA()))
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

/// factor are rewritten in terms of Q and d:
///   full solve chains F\(F'\b)  ->  Q diag(1/d) Q^T b
///   single solves F\b, F'\b    ->  Q diag(1/sqrt(d)) Q^T b
/// chol(B). Handles A + t*I (s = 1), a constant diagonal matrix folded by
/// recognized in every spelling of collectDiagonalExtractions.
///     factors c; every triangular_solve user is a
///     left-side solve with NO_TRANSPOSE or ADJOINT; at least one user is a
///     recognized solve chain, single solve, or diagonal extraction;
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

    // Chains first, so their inner solves are not taken as single solves.
    DenseSet<Operation *> innerSolveOps;
    for (auto *user : cholResult.getUsers()) {
      auto triSolve = dyn_cast<stablehlo::TriangularSolveOp>(user);
      if (!triSolve) {
        collectDiagonalExtractions(user, cholResult, match.diagExtractions);
        continue;
      }

      if (!triSolve.getLeftSide())
        return failure();

      if (triSolve.getTransposeA() == stablehlo::Transpose::NO_TRANSPOSE) {
        Value rhs = matchFullCholeskySolve(triSolve, cholResult);
        if (rhs) {
          auto innerSolve =
              triSolve.getB().getDefiningOp<stablehlo::TriangularSolveOp>();
          match.fullSolves.push_back({triSolve, innerSolve, rhs});
          innerSolveOps.insert(innerSolve.getOperation());
          continue;
        }
      }
    }

    for (auto *user : cholResult.getUsers()) {
      auto triSolve = dyn_cast<stablehlo::TriangularSolveOp>(user);
      if (!triSolve)
        continue;
      if (innerSolveOps.count(triSolve.getOperation()))
        continue;
      bool isFullSolveOuter = false;
      for (auto &fs : match.fullSolves) {
        if (fs.outerSolve == triSolve) {
          isFullSolveOuter = true;
          break;
        }
      }
      if (isFullSolveOuter)
        continue;

      auto transpose = triSolve.getTransposeA();
      if (transpose == stablehlo::Transpose::NO_TRANSPOSE) {
        match.singleSolves.push_back({triSolve, /*isAdjoint=*/false});
      } else if (isAdjointLikeTranspose(transpose)) {
        // ADJOINT or TRANSPOSE — equivalent for the real factors here.
        match.singleSolves.push_back({triSolve, /*isAdjoint=*/true});
      } else {
        return failure();
      }
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

// sum by an equal scalar computed from the hoistable statistics Q1, R, c, rem,
// dropping the per-gradient residual cost from O(N D) to O(D^2). The QR
// emission carries no large-magnitude cancellation and ships in the aggregate
// pattern set; the Gram emission (yty - 2 beta'Xty + beta'G beta) is the
// portability fallback behind an opt-in flag (Section 3.5/3.7 of
// GRAM_RULE_DESIGN.md).
enum class SicmAffineKind { Mul, Div, Add, SubRhs, SubLhs };
struct SicmAffineStep {
  SicmAffineKind kind;
  Value scalar;
};

struct GaussianStatsMatch {
  Value X, y, beta;
  int64_t N = 0, D = 0;
  int64_t xContractDim = 1;
  SmallVector<SicmAffineStep> affine; // outer-first
  bool hasFactor = false;
  Value factorScalar;
  bool factorIsDivide = false;
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
};

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
  return failure();
}

static FailureOr<GaussianStatsMatch>
matchGaussianSufficientStats(const SicmEventSum &es,
                             SampleDependenceAnalysis &analysis, int64_t cThr) {
  GaussianStatsMatch m;
  Value SQ = stripAffineScalarWrappers(es.K, m.affine);
  Value W = matchSquare(SQ);
  if (!W)
    return failure();
  Value R;
  m.hasFactor = peelScalarFactor(W, R, m.factorScalar, m.factorIsDivide);
  auto rm = matchGaussianResidual(R, analysis);
  if (failed(rm))
    return failure();
  if (rm->N != es.N)
    return failure();
  if (rm->D <= 0 || rm->N < cThr * rm->D)
    return failure();
  m.X = rm->X;
  m.y = rm->y;
  m.beta = rm->beta;
  m.N = rm->N;
  m.D = rm->D;
  m.xContractDim = rm->xContractDim;
  return m;
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
  auto [A, B] = composeAffine(b, loc, elemType, m.affine);
  Value Aterm = stablehlo::MulOp::create(b, loc, A, sumSQ);
  Value Nc = sicmScalarConst(b, loc, elemType, (double)m.N);
  Value NB = stablehlo::MulOp::create(b, loc, Nc, B);
  return stablehlo::AddOp::create(b, loc, Aterm, NB);
}

static RankedTensorType qrInfoType(OpBuilder &b) {
  return RankedTensorType::get({}, b.getIntegerType(32));
}

/// GaussianSufficientStats, QR form (default). Emits `enzymexla.linalg.qr X`
/// (the invariant Q1/R statistics hoist), computes c = Q1'y and the stable
/// remainder ||y - Q1 c||^2, and rewrites the in-loop residual as the D-vector
/// `c - R beta`. No cancellation, no numeric gate; registered in the aggregate
/// SicmPatterns set (Section 3.6/3.7).
static Value emitGaussianStatsQR(PatternRewriter &rewriter, Location loc,
                                 const GaussianStatsMatch &m) {
  Type elem = cast<RankedTensorType>(m.X.getType()).getElementType();
  Value X = getDesignND(rewriter, loc, m);

  auto qTy = RankedTensorType::get({m.N, m.D}, elem);
  auto rPackedTy = RankedTensorType::get({m.N, m.D}, elem);
  auto qr = enzymexla::QRFactorizationOp::create(
      rewriter, loc, TypeRange{qTy, rPackedTy, qrInfoType(rewriter)}, X);
  Value Q1 = qr.getQ(); // (N, D), orthonormal columns

  auto ddTy = RankedTensorType::get({m.D, m.D}, elem);
  auto dTy = RankedTensorType::get({m.D}, elem);
  auto nTy = RankedTensorType::get({m.N}, elem);
  auto scalarTy = RankedTensorType::get({}, elem);

  // Invariant statistics (hoisted by the fixpoint): R = Q1'X, c = Q1'y, and the
  // stable remainder rem = ||y - Q1 c||^2.
  Value R = emitContractDot(rewriter, loc, ddTy, Q1, X, {0}, {0});
  Value c = emitContractDot(rewriter, loc, dTy, Q1, m.y, {0}, {0});
  Value Q1c = emitContractDot(rewriter, loc, nTy, Q1, c, {1}, {0});
  Value resid = stablehlo::SubtractOp::create(rewriter, loc, m.y, Q1c);
  Value rem = emitContractDot(rewriter, loc, scalarTy, resid, resid, {0}, {0});

  Value Rb = emitContractDot(rewriter, loc, dTy, R, m.beta, {1}, {0});
  Value d = stablehlo::SubtractOp::create(rewriter, loc, c, Rb);
  Value dd = emitContractDot(rewriter, loc, scalarTy, d, d, {0}, {0});
  Value q = stablehlo::AddOp::create(rewriter, loc, dd, rem);
  return finishGaussianScalar(rewriter, loc, q, m, elem);
}

struct GaussianSufficientStatsHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  GaussianSufficientStatsHLO(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto es = matchEventSumDot(sumOp);
    if (failed(es))
      return failure();
    auto match = matchGaussianSufficientStats(*es, *analysis, /*cThr=*/2);
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
  GaussianSufficientStatsReduceHLO(MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(stablehlo::ReduceOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto es = matchEventSumReduce(sumOp);
    if (failed(es))
      return failure();
    auto match = matchGaussianSufficientStats(*es, *analysis, /*cThr=*/2);
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
                                   const GaussianStatsMatch &m) {
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
  return cast<RankedTensorType>(m.X.getType()).getElementType().isF64();
}

struct GaussianSufficientStatsGramHLO
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  GaussianSufficientStatsGramHLO(MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto es = matchEventSumDot(sumOp);
    if (failed(es))
      return failure();
    auto match = matchGaussianSufficientStats(*es, *analysis, /*cThr=*/2);
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
  GaussianSufficientStatsGramReduceHLO(MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(stablehlo::ReduceOp sumOp,
                                PatternRewriter &rewriter) const override {
    auto analysis = analyzeEnclosingRegion(sumOp);
    if (!analysis)
      return failure();
    auto es = matchEventSumReduce(sumOp);
    if (failed(es))
      return failure();
    auto match = matchGaussianSufficientStats(*es, *analysis, /*cThr=*/2);
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
      // Benefit 2 so the Gram form pre-empts the QR form on the same match.
      if (gramFormSufficientStats)
        patterns.add<GaussianSufficientStatsGramHLO,
                     GaussianSufficientStatsGramReduceHLO>(
            getOperation()->getContext(), PatternBenefit(2));
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
