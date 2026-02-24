#include "Dialect.h"

namespace mlir::enzyme::distributed {

LogicalResult
resolveLogicalAxisToAtomicFactors(Value logicalAxis,
                                  SmallVectorImpl<Value> &atomicFactors) {
  Operation *definingOp = logicalAxis.getDefiningOp();
  if (!definingOp)
    return failure();

  if (auto axisProduct = dyn_cast<AxisProductOp>(definingOp)) {
    for (Value operandAxis : axisProduct.getLogicalAxes()) {
      if (failed(resolveLogicalAxisToAtomicFactors(operandAxis, atomicFactors)))
        return failure();
    }
    return success();
  }

  if (isa<LogicalCommAxisOpInterface>(definingOp)) {
    atomicFactors.push_back(logicalAxis);
    return success();
  }

  return failure();
}

FailureOr<PhysicalCommAxisOpInterface>
resolvePhysicalAxisInterfaceFromAttr(Operation *from, Attribute axisAttr) {
  auto axisSymRef = dyn_cast_or_null<FlatSymbolRefAttr>(axisAttr);
  if (!axisSymRef) {
    from->emitOpError() << "requires a flat symbol ref to a physical axis";
    return failure();
  }

  Operation *axisOp = SymbolTable::lookupNearestSymbolFrom(from, axisSymRef);
  if (!axisOp) {
    from->emitOpError() << "references unknown physical axis symbol "
                        << axisSymRef;
    return failure();
  }

  auto axisInterface = dyn_cast<PhysicalCommAxisOpInterface>(axisOp);
  if (!axisInterface) {
    from->emitOpError() << "requires symbol to reference an op implementing "
                        << "PhysicalCommAxisOpInterface";
    return failure();
  }

  return axisInterface;
}

LogicalResult
resolveLogicalMeshToAtomicFactors(LogicalMeshOp logicalMesh,
                                  SmallVectorImpl<Value> &atomicFactors) {
  for (Value logicalAxis : logicalMesh.getAxes()) {
    if (failed(resolveLogicalAxisToAtomicFactors(logicalAxis, atomicFactors)))
      return failure();
  }
  return success();
}

bool isLogicalMeshDisjoint(LogicalMeshOp logicalMesh) {
  llvm::SmallVector<Value> atomicFactors;
  if (failed(resolveLogicalMeshToAtomicFactors(logicalMesh, atomicFactors))) {
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

bool isLogicalMeshSubmesh(LogicalMeshOp logicalMesh, LogicalMeshOp submesh) {
  llvm::SmallVector<Value> logicalMeshAtomicFactors;
  llvm::SmallVector<Value> submeshAtomicFactors;

  if (failed(resolveLogicalMeshToAtomicFactors(logicalMesh,
                                               logicalMeshAtomicFactors))) {
    return false;
  }
  if (failed(
          resolveLogicalMeshToAtomicFactors(submesh, submeshAtomicFactors))) {
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

FailureOr<int64_t> getLogicalMeshSize(LogicalMeshOp logicalMesh) {
  llvm::SmallVector<Value> atomicFactors;
  if (failed(resolveLogicalMeshToAtomicFactors(logicalMesh, atomicFactors))) {
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
