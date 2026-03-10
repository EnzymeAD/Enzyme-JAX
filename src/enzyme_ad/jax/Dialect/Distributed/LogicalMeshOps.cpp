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

LogicalResult LogicalMeshOp::resolveToAtomicFactors(
    SmallVectorImpl<TypedOpResult<LogicalCommAxisType>> &atomicFactors) {
  for (auto logicalAxis : getAxes()) {
    resolveLogicalAxisToAtomicFactors(logicalAxis, atomicFactors);
  }
  return success();
}

bool LogicalMeshOp::isDisjoint() {
  llvm::SmallVector<TypedOpResult<LogicalCommAxisType>> atomicFactors;
  if (failed(resolveToAtomicFactors(atomicFactors))) {
    return false;
  }

  llvm::SmallVector<SymbolRefAttr> physicalAxisRefs;
  for (auto atomicFactor : atomicFactors) {
    auto atomicFactorResult = atomicFactor.asOpResult();
    auto definingOp = atomicFactorResult.getDefiningOp();
    auto axisFactorOp = cast<AxisFactorOp>(definingOp);
    physicalAxisRefs.push_back(axisFactorOp.getPhysicalAxisAttr());
  }

  for (size_t i = 0; i < atomicFactors.size(); ++i) {
    auto factorIResult = atomicFactors[i].asOpResult();
    auto def_op = factorIResult.getDefiningOp();

    for (size_t j = i + 1; j < atomicFactors.size(); ++j) {
      auto factorJResult = atomicFactors[j].asOpResult();
      auto other_def_op = factorJResult.getDefiningOp();
      if (factorIResult == factorJResult) {
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
  llvm::SmallVector<TypedOpResult<LogicalCommAxisType>> logicalMeshAtomicFactors;
  llvm::SmallVector<TypedOpResult<LogicalCommAxisType>> submeshAtomicFactors;

  if (failed(resolveToAtomicFactors(logicalMeshAtomicFactors))) {
    return false;
  }
  if (failed(submesh.resolveToAtomicFactors(submeshAtomicFactors))) {
    return false;
  }

  if (submeshAtomicFactors.size() > logicalMeshAtomicFactors.size()) {
    return false;
  }

  for (auto submeshFactor : submeshAtomicFactors) {
    auto submeshFactorResult = submeshFactor.asOpResult();
    bool foundMatch = false;
    for (size_t i = 0; i < logicalMeshAtomicFactors.size(); ++i) {
      auto logicalMeshFactorResult = logicalMeshAtomicFactors[i].asOpResult();
      if (logicalMeshFactorResult == submeshFactorResult) {
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
  llvm::SmallVector<TypedOpResult<LogicalCommAxisType>> atomicFactors;
  if (failed(resolveToAtomicFactors(atomicFactors))) {
    return failure();
  }

  int64_t meshSize = 1;
  for (auto atomicFactor : atomicFactors) {
    auto atomicFactorResult = atomicFactor.asOpResult();
    auto definingOp = atomicFactorResult.getDefiningOp();
    auto axisInterface =
        dyn_cast_or_null<LogicalCommAxisOpInterface>(definingOp);
    if (!axisInterface) {
      return failure();
    }
    meshSize *= axisInterface.getAxisSize(atomicFactor);
  }

  return meshSize;
}

} // namespace mlir::enzyme::distributed
