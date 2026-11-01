#include "GrammarRegistry.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

namespace {

// A DFT computes each output digit from every input digit, so its
// transformed digits occur on one side only and must be local. An axis on
// both sides is a batch axis carried through.
BaseSignature dftSignature(OperatorAxesAttr axes, DictionaryAttr) {
  BaseSignature signature;
  for (int64_t axis = 0; axis < axes.getNumAxes(); ++axis) {
    auto occurrences = axes.getOccurrences(axis);
    bool onOperand = llvm::any_of(occurrences, [](const auto &occurrence) {
      return !occurrence.isResult;
    });
    bool onResult = llvm::any_of(occurrences, [](const auto &occurrence) {
      return occurrence.isResult;
    });
    signature.axes.push_back(onOperand && onResult ? AxisSignature::parallel()
                                                   : AxisSignature::local());
  }
  return signature;
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
  registry.registerEntry("dft", {dftSignature});
  registry.registerEntry("twiddle", {twiddleSignature});
  registry.registerRecognizer(recognizeComplexFft);
}

} // namespace mlir::enzyme::distributed
