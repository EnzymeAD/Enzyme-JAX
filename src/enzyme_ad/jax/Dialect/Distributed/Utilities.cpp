#include "Utilities.h"

namespace mlir::enzyme::distributed {

using axis::FactorGroupType;
using axis::getProductProvenanceFactors;

FailureOr<TypedValue<FactorGroupType>>
getEnclosingExecutionContext(Operation *op) {
  auto parentFunction = op ? op->getParentOfType<DistributedFunctionOp>()
                           : DistributedFunctionOp();
  if (!parentFunction) {
    return failure();
  }
  Value context = parentFunction.getExecutionContext();
  auto typedContext = dyn_cast<TypedValue<FactorGroupType>>(context);
  if (!typedContext) {
    return failure();
  }
  return typedContext;
}

FailureOr<llvm::SmallVector<Value>>
expandExecutionContextFactors(TypedValue<FactorGroupType> context) {
  FailureOr<ValueRange> factors = getProductProvenanceFactors(context);
  if (failed(factors)) {
    return failure();
  }
  llvm::SmallVector<Value> copiedFactors;
  copiedFactors.append(factors->begin(), factors->end());
  return copiedFactors;
}

} // namespace mlir::enzyme::distributed