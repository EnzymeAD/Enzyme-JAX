#include "Dialect.h"

#include "shardy/dialect/sdy/ir/utils.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::enzyme::distributed {

//===----------------------------------------------------------------------===//
// OperatorAxesAttr
//===----------------------------------------------------------------------===//

OperatorAxesAttr OperatorAxesAttr::get(
    MLIRContext *context, ArrayRef<int64_t> axisExtents,
    ArrayRef<SmallVector<SmallVector<int64_t>>> operandDigits,
    ArrayRef<SmallVector<SmallVector<int64_t>>> resultDigits) {
  auto toMappings = [&](ArrayRef<SmallVector<SmallVector<int64_t>>> digits) {
    return llvm::map_to_vector(
        digits, [&](ArrayRef<SmallVector<int64_t>> dims) {
          auto dimMappings =
              llvm::map_to_vector(dims, [&](ArrayRef<int64_t> axes) {
                return sdy::DimMappingAttr::get(context, axes);
              });
          return sdy::TensorMappingAttr::get(context, dimMappings);
        });
  };
  return get(context, axisExtents, toMappings(operandDigits),
             toMappings(resultDigits));
}

// Mirrors the per-tensor checks of Shardy's verifyShardingRuleMapping, which
// isn't exported. Checks against tensor types are GrammarTermOp's.
LogicalResult
OperatorAxesAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> axisExtents,
                         ArrayRef<sdy::TensorMappingAttr> operandMappings,
                         ArrayRef<sdy::TensorMappingAttr> resultMappings) {
  int64_t numAxes = axisExtents.size();
  for (auto [axis, extent] : llvm::enumerate(axisExtents)) {
    if (extent < 1)
      return emitError() << "operator axis " << sdy::factorSymbolString(axis)
                         << " has extent " << extent << ", expected at least 1";
  }

  llvm::BitVector usedAxes(numAxes);
  auto verifyMappings = [&](ArrayRef<sdy::TensorMappingAttr> mappings,
                            StringRef side) -> LogicalResult {
    for (auto [tensorIdx, mapping] : llvm::enumerate(mappings)) {
      llvm::BitVector usedByTensor(numAxes);
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        ArrayRef<int64_t> digits = dimMapping.getFactorIndices();
        for (int64_t axis : digits) {
          if (axis < 0 || axis >= numAxes)
            return emitError() << side << " " << tensorIdx << " dimension "
                               << dim << " maps operator axis index " << axis
                               << ", expected less than " << numAxes;
          if (usedByTensor.test(axis))
            return emitError()
                   << side << " " << tensorIdx << " maps operator axis "
                   << sdy::factorSymbolString(axis) << " more than once";
          if (digits.size() > 1 && axisExtents[axis] == 1)
            return emitError()
                   << side << " " << tensorIdx << " dimension " << dim
                   << " splits into several operator axes, but "
                   << sdy::factorSymbolString(axis) << " has extent 1";
          usedByTensor.set(axis);
        }
      }
      usedAxes |= usedByTensor;
    }
    return success();
  };
  if (failed(verifyMappings(operandMappings, "operand")) ||
      failed(verifyMappings(resultMappings, "result")))
    return failure();

  if (!usedAxes.all())
    return emitError() << "operator axis "
                       << sdy::factorSymbolString(usedAxes.find_first_unset())
                       << " is mapped by no operand or result";
  return success();
}

ArrayRef<int64_t> OperatorAxesAttr::getDigits(bool isResult, int64_t tensorIdx,
                                              int64_t dim) const {
  return getMappings(isResult)[tensorIdx]
      .getDimMappings()[dim]
      .getFactorIndices();
}

int64_t OperatorAxesAttr::getDigitStride(bool isResult, int64_t tensorIdx,
                                         int64_t dim, int64_t pos) const {
  int64_t stride = 1;
  for (int64_t axis : getDigits(isResult, tensorIdx, dim).drop_front(pos + 1))
    stride *= getAxisExtent(axis);
  return stride;
}

SmallVector<OperatorAxesAttr::Occurrence>
OperatorAxesAttr::getOccurrences(int64_t axis) const {
  SmallVector<Occurrence> occurrences;
  for (bool isResult : {false, true}) {
    for (auto [tensorIdx, mapping] : llvm::enumerate(getMappings(isResult))) {
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        for (auto [pos, digit] :
             llvm::enumerate(dimMapping.getFactorIndices())) {
          if (digit == axis)
            occurrences.push_back({isResult, static_cast<int64_t>(tensorIdx),
                                   static_cast<int64_t>(dim),
                                   static_cast<int64_t>(pos)});
        }
      }
    }
  }
  return occurrences;
}

//===----------------------------------------------------------------------===//
// GrammarTermOp
//===----------------------------------------------------------------------===//

void GrammarTermOp::build(
    OpBuilder &builder, OperationState &state, StringRef kind,
    ValueRange inputs, TypeRange outputTypes, ArrayRef<int64_t> axisExtents,
    ArrayRef<SmallVector<SmallVector<int64_t>>> operandDigits,
    ArrayRef<SmallVector<SmallVector<int64_t>>> resultDigits,
    DictionaryAttr params) {
  auto axes = OperatorAxesAttr::get(builder.getContext(), axisExtents,
                                    operandDigits, resultDigits);
  build(builder, state, outputTypes, inputs, kind, axes, params);
}

LogicalResult GrammarTermOp::verify() {
  if (getKind().empty())
    return emitOpError() << "requires a non-empty kind";

  OperatorAxesAttr axes = getAxes();
  auto verifyTensors = [&](TypeRange types, bool isResult) -> LogicalResult {
    StringRef side = isResult ? "result" : "operand";
    ArrayRef<sdy::TensorMappingAttr> mappings = axes.getMappings(isResult);
    if (mappings.size() != types.size())
      return emitOpError() << "has " << mappings.size() << " " << side
                           << " mappings for " << types.size() << " " << side
                           << "s";
    for (auto [tensorIdx, typeAndMapping] :
         llvm::enumerate(llvm::zip_equal(types, mappings))) {
      auto [type, mapping] = typeAndMapping;
      auto tensorType = cast<RankedTensorType>(type);
      if (mapping.getRank() != tensorType.getRank())
        return emitOpError()
               << side << " " << tensorIdx << " has rank "
               << tensorType.getRank() << " but its mapping has rank "
               << mapping.getRank();
      for (auto [dim, size] : llvm::enumerate(tensorType.getShape())) {
        int64_t product = 1;
        for (int64_t axis : axes.getDigits(isResult, tensorIdx, dim))
          product *= axes.getAxisExtent(axis);
        if (product != size)
          return emitOpError()
                 << side << " " << tensorIdx << " dimension " << dim
                 << " has size " << size
                 << " but its operator axes' extents multiply to " << product;
      }
    }
    return success();
  };
  if (failed(verifyTensors(getInputs().getTypes(), /*isResult=*/false)))
    return failure();
  return verifyTensors(getOutputs().getTypes(), /*isResult=*/true);
}

} // namespace mlir::enzyme::distributed
