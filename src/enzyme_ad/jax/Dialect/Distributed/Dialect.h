#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Base.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Dialect.h"

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

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
