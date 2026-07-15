#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

namespace {

static LogicalResult verifyFactorExtents(ArrayRef<int32_t> extents,
                                         unsigned sourceExtent,
                                         Operation *op) {
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

LogicalResult GetPhysicalAxisOp::verify() {
  FailureOr<PhysicalCommAxisOpInterface> physicalAxis =
      resolveSymbolOpFromAttr<PhysicalCommAxisOpInterface>(*this,
                                                           getPhysicalAxisAttr());
  if (failed(physicalAxis)) {
    return emitOpError() << "references unknown physical axis symbol "
                         << getPhysicalAxisAttr();
  }

  auto axisType = dyn_cast<PhysicalCommAxisType>(getAxis().getType());
  if (!axisType) {
    return emitOpError() << "requires result type to be PhysicalCommAxisType";
  }

  unsigned expectedExtent =
      static_cast<unsigned>(physicalAxis->getPhysicalAxisSize());
  if (axisType.getExtent() != expectedExtent) {
    return emitOpError() << "requires result type extent to match referenced "
                            "physical axis size ("
                         << axisType.getExtent() << " != " << expectedExtent
                         << ")";
  }

  return success();
}

LogicalResult GenericAxisOp::verify() {
  if (!isa<AxisTypeInterface>(getAxis().getType())) {
    return emitOpError() << "requires result type to implement AxisTypeInterface";
  }

  return success();
}

LogicalResult AxisFactorOp::verify() {
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
    if (axisFactorType.getExtent() != static_cast<unsigned>(factorExtents[idx])) {
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
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

} // namespace mlir::enzyme::distributed
