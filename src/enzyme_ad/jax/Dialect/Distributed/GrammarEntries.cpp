#include "GrammarRegistry.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::enzyme::distributed {

namespace {

// The axes that occur on both an input and an output. For a DFT such an axis
// is a batch axis carried through; any other axis is transformed.
llvm::BitVector getBothSidedAxes(OperatorAxesAttr roles) {
  llvm::BitVector onInput(roles.getNumAxes()), onOutput(roles.getNumAxes());
  for (bool isResult : {false, true})
    for (sdy::TensorMappingAttr mapping : roles.getMappings(isResult))
      for (sdy::DimMappingAttr dim : mapping.getDimMappings())
        for (int64_t axis : dim.getFactorIndices())
          (isResult ? onOutput : onInput).set(axis);
  onInput &= onOutput;
  return onInput;
}

// Whether `axes` holds some axis of `set` and some axis outside it.
bool mixes(ArrayRef<int64_t> axes, const llvm::BitVector &set) {
  return llvm::any_of(axes, [&](int64_t axis) { return set.test(axis); }) &&
         !llvm::all_of(axes, [&](int64_t axis) { return set.test(axis); });
}

// A DFT computes each output digit from every input digit, so its
// transformed digits occur on one side only and must be local. An axis on
// both sides is a batch axis carried through.
BaseSignature dftSignature(OperatorAxesAttr axes, DictionaryAttr) {
  llvm::BitVector bothSided = getBothSidedAxes(axes);
  BaseSignature signature;
  for (int64_t axis = 0; axis < axes.getNumAxes(); ++axis)
    signature.axes.push_back(bothSided.test(axis) ? AxisSignature::parallel()
                                                  : AxisSignature::local());
  return signature;
}

// A DFT's both-sided axes are pure batch, and their logical dimensions are
// dropped whole, along with dimensions that have no axes. dftVerify keeps
// each dimension either all both-sided or all one-sided, which makes that
// exact. On any other roles this stays total: a both-sided axis sharing a
// dimension with an axis that isn't peeled stays in the inner roles, so a
// dimension is either dropped or kept with all its axes.
PeeledBatch dftPeelBatch(OperatorAxesAttr roles, DictionaryAttr params) {
  auto forEachDim = [&](auto function) {
    for (bool isResult : {false, true})
      for (sdy::TensorMappingAttr mapping : roles.getMappings(isResult))
        for (sdy::DimMappingAttr dim : mapping.getDimMappings())
          function(dim.getFactorIndices());
  };
  // Unpeeling a mixed dimension's axes can mix another dimension, so this
  // repeats until no dimension mixes peeled and kept axes.
  llvm::BitVector peel = getBothSidedAxes(roles);
  bool changed = true;
  while (changed) {
    changed = false;
    forEachDim([&](ArrayRef<int64_t> axes) {
      if (!mixes(axes, peel))
        return;
      for (int64_t axis : axes)
        peel.reset(axis);
      changed = true;
    });
  }

  PeeledBatch peeled;
  peeled.innerParams = params;
  SmallVector<int64_t> innerAxisOf(roles.getNumAxes());
  SmallVector<int64_t> innerExtents;
  for (int64_t axis = 0; axis < roles.getNumAxes(); ++axis) {
    if (peel.test(axis)) {
      peeled.batchAxes.push_back(axis);
      continue;
    }
    innerAxisOf[axis] = peeled.innerAxes.size();
    peeled.innerAxes.push_back(axis);
    innerExtents.push_back(roles.getAxisExtent(axis));
  }

  using PositionDims = SmallVector<SmallVector<int64_t>>;
  auto innerPositionDims = [&](bool isResult) {
    SmallVector<PositionDims> positions;
    for (sdy::TensorMappingAttr mapping : roles.getMappings(isResult)) {
      PositionDims &dims = positions.emplace_back();
      for (sdy::DimMappingAttr dim : mapping.getDimMappings()) {
        ArrayRef<int64_t> axes = dim.getFactorIndices();
        if (llvm::all_of(axes, [&](int64_t axis) { return peel.test(axis); }))
          continue;
        SmallVector<int64_t> &innerDim = dims.emplace_back();
        for (int64_t axis : axes)
          innerDim.push_back(innerAxisOf[axis]);
      }
    }
    return positions;
  };
  SmallVector<PositionDims> inputs = innerPositionDims(/*isResult=*/false);
  SmallVector<PositionDims> outputs = innerPositionDims(/*isResult=*/true);
  peeled.innerRoles =
      OperatorAxesAttr::get(roles.getContext(), innerExtents, inputs, outputs);
  return peeled;
}

// A DFT dimension is either batch or transformed as a whole: a dimension
// mixing both-sided and one-sided axes has no meaning as a DFT.
LogicalResult dftVerify(GrammarTermOp term) {
  OperatorAxesAttr roles = term.getRoles();
  llvm::BitVector bothSided = getBothSidedAxes(roles);
  for (bool isResult : {false, true}) {
    for (auto [position, mapping] :
         llvm::enumerate(roles.getMappings(isResult))) {
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        if (mixes(dimMapping.getFactorIndices(), bothSided))
          return term.emitOpError()
                 << getTensorPositionName(isResult, position) << " dimension "
                 << dim
                 << " mixes axes on both sides with axes on one side, which "
                    "a dft doesn't allow";
      }
    }
  }
  return success();
}

// Twiddles scale each element by a constant of its global index, so every
// axis shards without communication.
BaseSignature twiddleSignature(OperatorAxesAttr axes, DictionaryAttr) {
  return BaseSignature{
      SmallVector<AxisSignature>(axes.getNumAxes(), AxisSignature::parallel())};
}

// A complex FFT or IFFT over its minor dimension is a DFT. The transformed
// dimension becomes an input axis and an output axis of the same extent; the
// other dimensions are batch axes shared by both sides. IFFT carries
// `inverse = true`.
std::optional<GrammarDescription> recognizeComplexFft(Operation *op) {
  auto fft = dyn_cast<stablehlo::FftOp>(op);
  if (!fft || fft.getFftLength().size() != 1)
    return std::nullopt;
  bool inverse = fft.getFftType() == stablehlo::FftType::IFFT;
  if (!inverse && fft.getFftType() != stablehlo::FftType::FFT)
    return std::nullopt;
  auto type = cast<RankedTensorType>(fft.getOperand().getType());
  if (!type.hasStaticShape())
    return std::nullopt;

  int64_t rank = type.getRank();
  SmallVector<int64_t> extents(type.getShape());
  extents.push_back(type.getShape().back());
  SmallVector<SmallVector<int64_t>> operandDims, resultDims;
  for (int64_t dim = 0; dim < rank; ++dim) {
    operandDims.push_back({dim});
    resultDims.push_back({dim});
  }
  resultDims.back() = {rank};

  MLIRContext *context = op->getContext();
  DictionaryAttr params;
  if (inverse)
    params = DictionaryAttr::get(
        context, {NamedAttribute(StringAttr::get(context, "inverse"),
                                 BoolAttr::get(context, true))});
  return GrammarDescription{
      StringAttr::get(context, "dft"),
      OperatorAxesAttr::get(context, extents, {operandDims}, {resultDims}),
      params, /*signature=*/{}};
}

} // namespace

void registerBuiltinGrammarEntries(GrammarRegistry &registry) {
  registry.registerEntry("dft", {dftSignature, dftPeelBatch, dftVerify});
  registry.registerEntry("twiddle", {twiddleSignature});
  registry.registerRecognizer(recognizeComplexFft);
}

} // namespace mlir::enzyme::distributed
