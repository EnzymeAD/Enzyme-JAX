#include "Dialect.h"

namespace mlir::enzyme::distributed {

void resolveLogicalAxisToAtomicFactors(
    TypedOpResult<LogicalCommAxisType> logicalAxis,
    SmallVectorImpl<TypedOpResult<LogicalCommAxisType>> &atomicFactors) {
  auto ax = logicalAxis.asOpResult();
  Operation *definingOp = ax.getDefiningOp();

  auto axisInterface = cast<LogicalCommAxisOpInterface>(definingOp);
  axisInterface.resolveToAtomicFactors(logicalAxis, atomicFactors);
}

int getAxisSize(TypedOpResult<LogicalCommAxisType> logicalAxis) {
  auto ax = logicalAxis.asOpResult();
  Operation *definingOp = ax.getDefiningOp();

  auto axisInterface = cast<LogicalCommAxisOpInterface>(definingOp);
  return axisInterface.getAxisSize(logicalAxis);
}
} // namespace mlir::enzyme::distributed
