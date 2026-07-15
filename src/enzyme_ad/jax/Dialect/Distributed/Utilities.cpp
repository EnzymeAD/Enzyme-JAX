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

FailureOr<PhysicalMeshOp> findUniquePhysicalMesh(ModuleOp moduleOp) {
  if (!moduleOp) {
    return failure();
  }

  unsigned physicalMeshCount = 0;
  PhysicalMeshOp physicalMesh;
  for (PhysicalMeshOp meshOp : moduleOp.getOps<PhysicalMeshOp>()) {
    ++physicalMeshCount;
    if (physicalMeshCount == 1) {
      physicalMesh = meshOp;
    }
    if (physicalMeshCount > 1) {
      moduleOp.emitError()
          << "expected exactly one distributed physical mesh in module, found "
          << physicalMeshCount;
      return failure();
    }
  }

  if (physicalMeshCount == 0) {
    moduleOp.emitError()
        << "expected exactly one distributed physical mesh in module, found 0";
    return failure();
  }

  return physicalMesh;
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

::llvm::SmallVector<TypedValue<::mlir::enzyme::axis::AxisFactorType>>
filterOutReplicationFactors(TypedValueArrayRef<::mlir::enzyme::axis::AxisFactorType> factors) {
  llvm::SmallVector<TypedValue<::mlir::enzyme::axis::AxisFactorType>> filteredFactors;
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