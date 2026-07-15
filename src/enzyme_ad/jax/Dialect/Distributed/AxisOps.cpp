#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

LogicalResult PhysicalMeshOp::verify() {
  for (auto [idx, axisAttr] : llvm::enumerate(getAxesAttr())) {
    auto typeAttr = dyn_cast<TypeAttr>(axisAttr);
    if (!typeAttr) {
      return emitOpError() << "requires axes[" << idx << "] to be a TypeAttr";
    }
    if (!isa<PhysicalCommAxisType>(typeAttr.getValue())) {
      return emitOpError() << "requires axes[" << idx
                           << "] to be a PhysicalCommAxisType attribute";
    }
  }

  return success();
}

LogicalResult GetPhysicalMeshAxesOp::verify() {
  FailureOr<PhysicalMeshOp> physicalMesh =
      resolveSymbolOpFromAttr<PhysicalMeshOp>(*this, getPhysicalMeshAttr());
  if (failed(physicalMesh)) {
    return emitOpError() << "references unknown physical mesh symbol "
                         << getPhysicalMeshAttr();
  }

  ArrayAttr axisAttrs = physicalMesh->getAxesAttr();
  if (getAxes().size() != axisAttrs.size()) {
    return emitOpError() << "requires result count to match referenced "
                            "physical mesh axes size ("
                         << getAxes().size() << " != " << axisAttrs.size()
                         << ")";
  }

  for (auto [idx, axisAttr] : llvm::enumerate(axisAttrs)) {
    auto typeAttr = dyn_cast<TypeAttr>(axisAttr);
    if (!typeAttr) {
      return emitOpError() << "requires referenced physical mesh axes[" << idx
                           << "] to be a TypeAttr";
    }

    auto expectedAxisType = dyn_cast<PhysicalCommAxisType>(typeAttr.getValue());
    if (!expectedAxisType) {
      return emitOpError() << "requires referenced physical mesh axes[" << idx
                           << "] to be a PhysicalCommAxisType attribute";
    }

    auto actualAxisType =
        dyn_cast<PhysicalCommAxisType>(getAxes()[idx].getType());
    if (!actualAxisType) {
      return emitOpError() << "requires result #" << idx
                           << " to have PhysicalCommAxisType";
    }

    if (actualAxisType != expectedAxisType) {
      return emitOpError() << "requires result #" << idx << " to have type "
                           << expectedAxisType << ", but got "
                           << actualAxisType;
    }
  }

  return success();
}

LogicalResult LogicalMeshAxesOp::verify() {
  ArrayRef<int32_t> axisExtents = getAxisExtents();
  if (getAxes().size() != axisExtents.size()) {
    return emitOpError() << "requires number of results to match number of "
                            "axis_extents ("
                         << getAxes().size() << " != " << axisExtents.size()
                         << ")";
  }

  for (auto [idx, axisExtent] : llvm::enumerate(axisExtents)) {
    if (axisExtent <= 0) {
      return emitOpError() << "requires axis_extents[" << idx
                           << "] to be positive, got " << axisExtent;
    }

    auto actualAxisType =
        dyn_cast<LogicalMeshAxisType>(getAxes()[idx].getType());
    if (!actualAxisType) {
      return emitOpError() << "requires result #" << idx
                           << " to have LogicalMeshAxisType";
    }

    if (actualAxisType.getExtent() != static_cast<unsigned>(axisExtent)) {
      return emitOpError() << "requires result #" << idx << " to have extent "
                           << axisExtent << ", but got "
                           << actualAxisType.getExtent();
    }
  }

  return success();
}

LogicalResult LogicalMeshAxesOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  LogicalMeshAxesOpAdaptor adaptor(operands, attributes, properties, regions);

  ArrayRef<int32_t> axisExtents = adaptor.getAxisExtents();
  inferredReturnTypes.reserve(axisExtents.size());
  for (auto [idx, axisExtent] : llvm::enumerate(axisExtents)) {
    if (axisExtent <= 0) {
      if (location) {
        mlir::emitError(*location) << "requires axis_extents[" << idx
                                   << "] to be positive, got " << axisExtent;
      }
      return failure();
    }

    inferredReturnTypes.push_back(
        LogicalMeshAxisType::get(context, static_cast<unsigned>(axisExtent)));
  }

  return success();
}

} // namespace mlir::enzyme::distributed
