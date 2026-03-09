#include "Dialect.h"

namespace mlir::enzyme::distributed {

void
resolveLogicalAxisToAtomicFactors(Value logicalAxis,
                                  SmallVectorImpl<Value> &atomicFactors) {
  Operation *definingOp = logicalAxis.getDefiningOp();
  if (!definingOp) {
    emitError(logicalAxis.getLoc())
        << "logical axis must be defined by an op implementing "
        << "LogicalCommAxisOpInterface";
      return;
  }

  auto axisInterface = dyn_cast<LogicalCommAxisOpInterface>(definingOp);
  if (!axisInterface) {
    emitError(logicalAxis.getLoc())
        << "logical axis is defined by op '"
        << definingOp->getName().getStringRef()
        << "' which does not implement LogicalCommAxisOpInterface";
    return;
  }

  axisInterface.resolveToAtomicFactors(logicalAxis, atomicFactors);
}

} // namespace mlir::enzyme::distributed
