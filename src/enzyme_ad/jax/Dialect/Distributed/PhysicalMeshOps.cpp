#include "Dialect.h"

namespace mlir::enzyme::distributed {

LogicalResult PhysicalMeshOp::verify() {
  for (auto axisRef : getAxes()) {
    if (failed(resolveSymbolOpFromAttr<PhysicalCommAxisOpInterface>(*this,
                                                                    axisRef))) {
      return emitOpError() << "references unknown physical axis symbol "
                           << axisRef;
    }
  }
  return mlir::success();
}

unsigned PhysicalMeshOp::getPhysicalAxisPosition(FlatSymbolRefAttr axis) {
  auto axes = getAxes();
  for (unsigned i = 0; i < axes.size(); ++i) {
    if (axes[i] == axis) {
      return i;
    }
  }
  assert(false && "factor physical axis must be present in mesh axes");
  return 0;
}

} // namespace mlir::enzyme::distributed
