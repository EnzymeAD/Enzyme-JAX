#include "GrammarRegistry.h"

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

} // namespace

void registerBuiltinGrammarEntries(GrammarRegistry &registry) {
  registry.registerEntry("dft", {dftSignature});
  registry.registerEntry("twiddle", {twiddleSignature});
}

} // namespace mlir::enzyme::distributed
