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
      return emitOpError()
             << "requires referenced physical mesh axes[" << idx
             << "] to be a TypeAttr";
    }

    auto expectedAxisType = dyn_cast<PhysicalCommAxisType>(typeAttr.getValue());
    if (!expectedAxisType) {
      return emitOpError()
             << "requires referenced physical mesh axes[" << idx
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

} // namespace mlir::enzyme::distributed
