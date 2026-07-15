#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H

#include "Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

namespace mlir::enzyme::distributed {

template <typename OpTy>
::mlir::FailureOr<OpTy> resolveSymbolOpFromAttr(::mlir::Operation *from,
                                                ::mlir::Attribute opAttr) {
  auto symRef = ::mlir::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(opAttr);
  if (!symRef) {
    return ::mlir::failure();
  }
  auto *op = ::mlir::SymbolTable::lookupNearestSymbolFrom(from, symRef);
  if (!op) {
    return ::mlir::failure();
  }
  auto typedOp = llvm::dyn_cast<OpTy>(op);
  if (!typedOp) {
    return ::mlir::failure();
  }
  return typedOp;
}

// Returns the execution-context FactorGroup value from the nearest enclosing
// distributed.function.
::mlir::FailureOr<::mlir::TypedValue<::mlir::enzyme::axis::FactorGroupType>>
getEnclosingExecutionContext(::mlir::Operation *op);

// Expands an axis.factor_group value into its defining axis.factor list.
::mlir::FailureOr<::llvm::SmallVector<::mlir::Value>>
expandExecutionContextFactors(
    ::mlir::TypedValue<::mlir::enzyme::axis::FactorGroupType> context);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
