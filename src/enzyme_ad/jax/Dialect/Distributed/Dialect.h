#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

#include "shardy/dialect/sdy/ir/dialect.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedDialect.h.inc"
// Traits and interfaces
#include "Traits.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedInterfaces.h.inc"
// Types
#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.h.inc"
// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.h.inc"

// Utilities
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

/**
 * Decomposes a logical axis into the SSA values resulting from
 * the `factor` calls on a physical axis.
 */
void resolveLogicalAxisToAtomicFactors(
    ::mlir::Value logicalAxis,
    ::llvm::SmallVectorImpl<::mlir::Value> &atomicFactors);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
