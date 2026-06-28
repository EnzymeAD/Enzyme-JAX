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

using ::mlir::enzyme::axis::areFactorsComplete;
using ::mlir::enzyme::axis::areFactorsDisjoint;
using ::mlir::enzyme::axis::areSegmentsComplete;
using ::mlir::enzyme::axis::areSegmentsDisjoint;
using ::mlir::enzyme::axis::getAxisExtent;
using ::mlir::enzyme::axis::getFactorExtent;
using ::mlir::enzyme::axis::getFactorProvenanceAxis;
using ::mlir::enzyme::axis::getSegmentExtent;
using ::mlir::enzyme::axis::getSegmentProvenanceAxis;

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
