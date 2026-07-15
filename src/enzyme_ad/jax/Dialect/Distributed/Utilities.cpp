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

} // namespace mlir::enzyme::distributed
