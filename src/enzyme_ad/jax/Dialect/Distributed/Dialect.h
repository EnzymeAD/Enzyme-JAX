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

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
