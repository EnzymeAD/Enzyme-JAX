#include "Utilities.h"

namespace mlir::enzyme::distributed {

using namespace ::mlir::enzyme::axis;

Operation *lookupSymbolInEnclosingScopes(Operation *from,
                                         FlatSymbolRefAttr symRef) {
  if (!from || !symRef) {
    return nullptr;
  }

  for (auto *scope = from; scope; scope = scope->getParentOp()) {
    if (!scope->hasTrait<OpTrait::SymbolTable>()) {
      continue;
    }
    if (auto *op = SymbolTable::lookupSymbolIn(scope, symRef)) {
      return op;
    }
  }

  return nullptr;
}

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

::llvm::SmallVector<::mlir::Value>
filterOutReplicationFactors(::mlir::ValueRange factors) {
  llvm::SmallVector<::mlir::Value> filteredFactors;
  for (auto factor : factors) {
    // type of factor should wrap replication axis if it is a replication factor
    auto factorType =
        cast<::mlir::enzyme::axis::AxisFactorType>(factor.getType());
    ::mlir::Type axisType = factorType.getAxisType();
    if (!isa<::mlir::enzyme::distributed::ReplicationAxisType>(axisType)) {
      filteredFactors.push_back(factor);
    }
  }
  return filteredFactors;
}
} // namespace mlir::enzyme::distributed