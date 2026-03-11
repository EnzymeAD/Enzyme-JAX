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

// Need to declare the TypedOpResult struct before including the generated
// interface declarations
namespace mlir::enzyme::distributed {
template <typename TypeTy> struct TypedOpResult {
  ::mlir::OpResult value;
  TypedOpResult(mlir::Value value) : value(llvm::cast<mlir::OpResult>(value)) {
    assert(isa<TypeTy>(value.getType()) && "value must have the correct type");
  }
  TypedOpResult(::mlir::OpResult value) : value(value) {
    assert(isa<TypeTy>(value.getType()) && "value must have the correct type");
  }
  operator mlir::OpResult() const { return value; }

  mlir::OpResult asOpResult() const { return value; }
};
} // namespace mlir::enzyme::distributed

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedDialect.h.inc"
// Traits
#include "Traits.h"
// Types
#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.h.inc"
// Interfaces
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedInterfaces.h.inc"
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
    TypedOpResult<LogicalCommAxisType> logicalAxis,
    ::llvm::SmallVectorImpl<TypedOpResult<LogicalCommAxisType>> &atomicFactors);

TypedOpResult<CollectiveTokenType> resolveCollectiveTokenToRootCollective(
    TypedOpResult<CollectiveTokenType> token);

int getAxisSize(TypedOpResult<LogicalCommAxisType> logicalAxis);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
