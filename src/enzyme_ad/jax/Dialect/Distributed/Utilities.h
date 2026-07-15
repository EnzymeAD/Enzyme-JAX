#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H

#include "Dialect.h"

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

// Returns the static extent for any axis-typed SSA value.
int getAxisSize(::mlir::Value axis);

// Returns the static extent for any factor-typed SSA value.
int getFactorSize(::mlir::Value factor);

// Resolves the source axis used to produce a factor value.
::mlir::FailureOr<::mlir::Value> getFactorProvenanceAxis(::mlir::Value factor);

// Checks that factors are pairwise non-overlapping for one source axis.
// Replication factors are treated as always disjoint.
bool areFactorsDisjoint(::mlir::ValueRange factors);

// Checks that factors cover an axis exactly and therefore are disjoint.
// Replication axes/factors are ignored by this check.
bool areFactorsComplete(::mlir::Value axis, ::mlir::ValueRange factors);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
