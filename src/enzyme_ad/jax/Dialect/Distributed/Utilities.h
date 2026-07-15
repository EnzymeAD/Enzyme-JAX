#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H

#include "Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

namespace mlir::enzyme::distributed {

using ::mlir::enzyme::axis::castTypedValue;
using ::mlir::enzyme::axis::castTypedValueList; 
using ::mlir::enzyme::axis::TypedValueArrayRef;

// Walks parent operations and checks each symbol table scope for a flat symbol.
::mlir::Operation *
lookupSymbolInEnclosingScopes(::mlir::Operation *from,
                              ::mlir::FlatSymbolRefAttr symRef);

// Finds the unique distributed physical mesh in the module.
::mlir::FailureOr<::mlir::enzyme::distributed::PhysicalMeshOp>
findUniquePhysicalMesh(::mlir::ModuleOp moduleOp);

template <typename OpTy>
::mlir::FailureOr<OpTy> resolveSymbolOpFromAttr(::mlir::Operation *from,
                                                ::mlir::Attribute opAttr) {
  auto symRef = ::mlir::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(opAttr);
  if (!symRef) {
    return ::mlir::failure();
  }

  if (auto *op = lookupSymbolInEnclosingScopes(from, symRef)) {
    if (auto typedOp = llvm::dyn_cast<OpTy>(op)) {
      return typedOp;
    }
    return ::mlir::failure();
  }

  return ::mlir::failure();
}

// Returns the execution-context FactorGroup value from the nearest enclosing
// distributed.function.
::mlir::FailureOr<::mlir::TypedValue<::mlir::enzyme::axis::FactorGroupType>>
getEnclosingExecutionContext(::mlir::Operation *op);

// Creates a new range with all replication axes removed from the input range.
::llvm::SmallVector<TypedValue<::mlir::enzyme::axis::AxisFactorType>>
filterOutReplicationFactors(TypedValueArrayRef<::mlir::enzyme::axis::AxisFactorType> factors);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
