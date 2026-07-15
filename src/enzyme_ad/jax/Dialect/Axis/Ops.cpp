#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::axis {

namespace {

static LogicalResult verifyFactorExtents(ArrayRef<int32_t> extents,
                                         unsigned sourceExtent, Operation *op) {
  uint64_t product = 1;
  for (int32_t extent : extents) {
    if (extent <= 0) {
      return op->emitOpError() << "requires all factor extents to be > 0";
    }
    product *= static_cast<uint64_t>(extent);
  }
  if (product != sourceExtent) {
    return op->emitOpError() << "requires product(factor_extents) == axis "
                                "extent ("
                             << product << " != " << sourceExtent << ")";
  }
  return success();
}

static LogicalResult verifySegmentExtents(ArrayRef<int32_t> extents,
                                          unsigned sourceExtent,
                                          Operation *op) {
  uint64_t sum = 0;
  for (int32_t extent : extents) {
    if (extent <= 0) {
      return op->emitOpError() << "requires all segment extents to be > 0";
    }
    sum += static_cast<uint64_t>(extent);
  }
  if (sum != sourceExtent) {
    return op->emitOpError() << "requires sum(segment_extents) == axis extent "
                                "("
                             << sum << " != " << sourceExtent << ")";
  }
  return success();
}

static void computeMajorToMinorStrides(ArrayRef<int32_t> extents,
                                       SmallVectorImpl<unsigned> &strides) {
  // Leftmost factors are most major.
  strides.resize(extents.size());
  unsigned running = 1;
  for (int i = static_cast<int>(extents.size()) - 1; i >= 0; --i) {
    strides[i] = running;
    running *= static_cast<unsigned>(extents[i]);
  }
}

} // namespace

LogicalResult AxisGetAxisOp::verify() {
  auto shaped = dyn_cast<ShapedType>(getShapeType());
  if (!shaped) {
    return emitOpError() << "requires shape_type to be a shaped type";
  }
  if (!shaped.hasRank()) {
    return emitOpError() << "requires shape_type to be ranked";
  }

  int64_t axisIndex = getAxisIndex();
  if (axisIndex < 0 || axisIndex >= shaped.getRank()) {
    return emitOpError() << "requires axis_index in [0, rank), got "
                         << axisIndex << " for rank " << shaped.getRank();
  }
  if (shaped.isDynamicDim(static_cast<unsigned>(axisIndex))) {
    return emitOpError() << "requires static shape dimension at axis_index "
                         << axisIndex;
  }

  auto expectedType = ShapeAxisType::get(getContext(), getShapeType(),
                                         static_cast<unsigned>(axisIndex));
  if (getAxis().getType() != expectedType) {
    return emitOpError()
           << "requires result type to match shape_type and axis_index";
  }

  return success();
}

LogicalResult AxisGetAxisOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AxisGetAxisOpAdaptor adaptor(operands, attributes, properties, regions);
  int64_t axisIndex = adaptor.getAxisIndex();
  if (axisIndex < 0) {
    if (location) {
      mlir::emitError(*location) << "requires non-negative axis_index";
    }
    return failure();
  }

  inferredReturnTypes.push_back(ShapeAxisType::get(
      context, adaptor.getShapeType(), static_cast<unsigned>(axisIndex)));
  return success();
}

LogicalResult AxisFactorOp::verify() {
  if (failed(getAxisProvenanceOp(getAxis()))) {
    return emitOpError()
           << "requires axis operand to be traceable to an op result";
  }

  auto axisIface = dyn_cast<AxisTypeInterface>(getAxis().getType());
  if (!axisIface) {
    return emitOpError() << "requires axis operand type to implement "
                            "AxisTypeInterface";
  }

  ArrayRef<int32_t> factorExtents = getFactorExtents();

  if (failed(verifyFactorExtents(factorExtents, axisIface.extent(),
                                 getOperation()))) {
    return failure();
  }

  if (getAxisFactors().size() != factorExtents.size()) {
    return emitOpError() << "requires number of results to match number of "
                            "factor extents";
  }

  SmallVector<unsigned> expectedStrides;
  computeMajorToMinorStrides(factorExtents, expectedStrides);

  for (auto [idx, axisFactorVal] : llvm::enumerate(getAxisFactors())) {
    auto axisFactorType = dyn_cast<AxisFactorType>(axisFactorVal.getType());
    if (!axisFactorType) {
      return emitOpError() << "requires all results to have AxisFactorType";
    }
    if (axisFactorType.getAxisType() != getAxis().getType()) {
      return emitOpError() << "requires result #" << idx
                           << " to have base axis type equal to operand type";
    }
    if (axisFactorType.getExtent() !=
        static_cast<unsigned>(factorExtents[idx])) {
      return emitOpError() << "requires result #" << idx
                           << " extent to match factor extent";
    }
    if (axisFactorType.getStride() != expectedStrides[idx]) {
      return emitOpError() << "requires result #" << idx
                           << " stride to follow leftmost-major convention";
    }
  }

  return success();
}

LogicalResult AxisFactorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AxisFactorOpAdaptor adaptor(operands, attributes, properties, regions);
  ArrayRef<int32_t> factorExtents = adaptor.getFactorExtents();

  SmallVector<unsigned> strides;
  computeMajorToMinorStrides(factorExtents, strides);

  inferredReturnTypes.reserve(factorExtents.size());
  Type axisType = adaptor.getAxis().getType();
  for (auto [idx, factorExtent] : llvm::enumerate(factorExtents)) {
    inferredReturnTypes.push_back(AxisFactorType::get(
        context, axisType, static_cast<unsigned>(factorExtent), strides[idx]));
  }
  return success();
}

LogicalResult AxisSegmentOp::verify() {
  if (failed(getAxisProvenanceOp(getAxis()))) {
    return emitOpError()
           << "requires axis operand to be traceable to an op result";
  }

  auto axisIface = dyn_cast<AxisTypeInterface>(getAxis().getType());
  if (!axisIface) {
    return emitOpError() << "requires axis operand type to implement "
                            "AxisTypeInterface";
  }

  ArrayRef<int32_t> segmentExtents = getSegmentExtents();
  if (failed(verifySegmentExtents(segmentExtents, axisIface.extent(),
                                  getOperation()))) {
    return failure();
  }

  if (getAxisSegments().size() != segmentExtents.size()) {
    return emitOpError() << "requires number of results to match number of "
                            "segment extents";
  }

  unsigned runningOffset = 0;
  for (auto [idx, axisSegmentVal] : llvm::enumerate(getAxisSegments())) {
    auto axisSegmentType = dyn_cast<AxisSegmentType>(axisSegmentVal.getType());
    if (!axisSegmentType) {
      return emitOpError() << "requires all results to have AxisSegmentType";
    }
    if (axisSegmentType.getAxisType() != getAxis().getType()) {
      return emitOpError() << "requires result #" << idx
                           << " to have base axis type equal to operand type";
    }
    if (axisSegmentType.getExtent() !=
        static_cast<unsigned>(segmentExtents[idx])) {
      return emitOpError() << "requires result #" << idx
                           << " extent to match segment extent";
    }
    if (axisSegmentType.getOffset() != runningOffset) {
      return emitOpError() << "requires result #" << idx
                           << " offset to match cumulative segment layout "
                              "(low result index maps to low axis values)";
    }
    runningOffset += static_cast<unsigned>(segmentExtents[idx]);
  }

  return success();
}

LogicalResult AxisSegmentOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AxisSegmentOpAdaptor adaptor(operands, attributes, properties, regions);
  ArrayRef<int32_t> segmentExtents = adaptor.getSegmentExtents();

  inferredReturnTypes.reserve(segmentExtents.size());
  Type axisType = adaptor.getAxis().getType();
  unsigned runningOffset = 0;
  for (int32_t segmentExtent : segmentExtents) {
    inferredReturnTypes.push_back(AxisSegmentType::get(
        context, axisType, static_cast<unsigned>(segmentExtent),
        runningOffset));
    runningOffset += static_cast<unsigned>(segmentExtent);
  }
  return success();
}

LogicalResult AxisProductOp::verify() {
  uint64_t extentProduct = 1;
  for (Value factor : getFactors()) {
    if (!isa<OpResult>(factor)) {
      return emitOpError() << "requires factor operands to be op results";
    }
    if (!factor.getDefiningOp<AxisFactorOp>()) {
      return emitOpError()
             << "requires factor operands to be produced by axis.factor";
    }

    auto factorType = dyn_cast<AxisFactorType>(factor.getType());
    if (!factorType) {
      return emitOpError() << "requires all operands to be AxisFactorType";
    }
    extentProduct *= static_cast<uint64_t>(factorType.getExtent());
  }

  if (getProduct().getType().getExtent() != extentProduct) {
    return emitOpError()
           << "requires product extent to equal product of factor extents";
  }
  return success();
}

LogicalResult AxisProductOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AxisProductOpAdaptor adaptor(operands, attributes, properties, regions);
  uint64_t extentProduct = 1;
  for (Value factor : adaptor.getFactors()) {
    auto factorType = dyn_cast<AxisFactorType>(factor.getType());
    if (!factorType) {
      if (location)
        mlir::emitError(*location)
            << "requires all operands to be AxisFactorType";
      return failure();
    }
    extentProduct *= static_cast<uint64_t>(factorType.getExtent());
  }

  inferredReturnTypes.push_back(
      FactorGroupType::get(context, static_cast<unsigned>(extentProduct)));
  return success();
}

} // namespace mlir::enzyme::axis

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisOps.cpp.inc"
