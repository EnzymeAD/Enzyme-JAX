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

#include "Traits.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.h.inc"

#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedInterfaces.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.h.inc"

/**
 * Convenience class to manage tokens, which are sometimes used  as
 * block args and other time as typed values.
 */
namespace mlir::enzyme::distributed {
class Token {
  mlir::TypedValue<TokenType> typedValue;
  mlir::BlockArgument blockArg;

public:
  Token(mlir::BlockArgument arg) : blockArg(arg) {
    typedValue = dyn_cast<mlir::TypedValue<TokenType>>(arg);
    assert(typedValue && "Block arg is not a token");
  }
  Token(mlir::TypedValue<TokenType> val) : typedValue(val) {
    assert(val && "Typed value is null");
    blockArg = dyn_cast<mlir::BlockArgument>(val);
    assert(blockArg && "Typed value is not a block argument");
  }

  const mlir::TypedValue<TokenType> asTypedValue() const { return typedValue; }
  const mlir::BlockArgument asBlockArg() const { return blockArg; }
};
} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
