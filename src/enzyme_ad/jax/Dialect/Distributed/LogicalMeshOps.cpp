#include "Dialect.h"

namespace mlir::enzyme::distributed {

LogicalResult LogicalMeshOp::verify() {
  if (!isDisjoint()) {
    return emitOpError()
           << "requires mesh factors to be disjoint, and all factors for the "
           << "same physical axis to come from a single factorization op";
  }
  return mlir::success();
}

FailureOr<PhysicalMeshOp> LogicalMeshOp::resolvePhysicalMesh() {
  auto physicalMeshOr = resolveSymbolOpFromAttr<PhysicalMeshOp>(
      getOperation(), getPhysicalMeshAttr());
  if (failed(physicalMeshOr)) {
    emitOpError() << "requires physical_mesh to reference a PhysicalMesh op";
    return failure();
  }
  return *physicalMeshOr;
}

LogicalResult
LogicalMeshOp::resolveToAtomicFactors(SmallVectorImpl<Value> &atomicFactors) {
  for (Value logicalAxis : getAxes()) {
    resolveLogicalAxisToAtomicFactors(logicalAxis, atomicFactors);
  }
  return success();
}

bool LogicalMeshOp::isDisjoint() {
  llvm::SmallVector<Value> atomicFactors;
  if (failed(resolveToAtomicFactors(atomicFactors))) {
    return false;
  }

  llvm::SmallVector<SymbolRefAttr> physicalAxisRefs;
  for (Value atomicFactor : atomicFactors) {
    auto definingOp = atomicFactor.getDefiningOp();
    auto axisFactorOp = cast<AxisFactorOp>(definingOp);
    physicalAxisRefs.push_back(axisFactorOp.getPhysicalAxisAttr());
  }

  for (size_t i = 0; i < atomicFactors.size(); ++i) {
    auto def_op = atomicFactors[i].getDefiningOp();

    for (size_t j = i + 1; j < atomicFactors.size(); ++j) {
      auto other_def_op = atomicFactors[j].getDefiningOp();
      if (atomicFactors[i] == atomicFactors[j]) {
        return false;
      }
      if (physicalAxisRefs[i] == physicalAxisRefs[j] &&
          def_op != other_def_op) {
        return false;
      }
    }
  }

  return true;
}

bool LogicalMeshOp::isSubmesh(LogicalMeshOp submesh) {
  llvm::SmallVector<Value> logicalMeshAtomicFactors;
  llvm::SmallVector<Value> submeshAtomicFactors;

  if (failed(resolveToAtomicFactors(logicalMeshAtomicFactors))) {
    return false;
  }
  if (failed(submesh.resolveToAtomicFactors(submeshAtomicFactors))) {
    return false;
  }

  if (submeshAtomicFactors.size() > logicalMeshAtomicFactors.size()) {
    return false;
  }

  for (Value submeshFactor : submeshAtomicFactors) {
    bool foundMatch = false;
    for (size_t i = 0; i < logicalMeshAtomicFactors.size(); ++i) {
      if (logicalMeshAtomicFactors[i] == submeshFactor) {
        foundMatch = true;
        break;
      }
    }
    if (!foundMatch) {
      return false;
    }
  }

  return true;
}

FailureOr<int64_t> LogicalMeshOp::getMeshSize() {
  llvm::SmallVector<Value> atomicFactors;
  if (failed(resolveToAtomicFactors(atomicFactors))) {
    return failure();
  }

  int64_t meshSize = 1;
  for (Value atomicFactor : atomicFactors) {
    auto definingOp = atomicFactor.getDefiningOp();
    auto axisInterface = dyn_cast_or_null<LogicalCommAxisOpInterface>(definingOp);
    if (!axisInterface) {
      return failure();
    }
    meshSize *= axisInterface.getAxisSize(atomicFactor);
  }

  return meshSize;
}

} // namespace mlir::enzyme::distributed
