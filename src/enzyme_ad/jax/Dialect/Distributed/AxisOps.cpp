#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

LogicalResult GetPhysicalAxisOp::verify() {
  FailureOr<PhysicalCommAxisOpInterface> physicalAxis =
      resolveSymbolOpFromAttr<PhysicalCommAxisOpInterface>(
          *this, getPhysicalAxisAttr());
  if (failed(physicalAxis)) {
    return emitOpError() << "references unknown physical axis symbol "
                         << getPhysicalAxisAttr();
  }

  unsigned expectedExtent =
      static_cast<unsigned>(physicalAxis->getPhysicalAxisExtent());
  auto axisType = getAxis().getType();
  if (axisType.getExtent() != expectedExtent) {
    return emitOpError() << "requires result type extent to match referenced "
                            "physical axis extent ("
                         << axisType.getExtent() << " != " << expectedExtent
                         << ")";
  }

  return success();
}

} // namespace mlir::enzyme::distributed
