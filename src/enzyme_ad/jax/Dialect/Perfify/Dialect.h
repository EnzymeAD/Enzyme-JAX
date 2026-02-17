#ifndef ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyDialect.h.inc"

// Types
#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyTypes.h.inc"
// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.h.inc"

#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_H