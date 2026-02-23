#ifndef ENZYMEXLA_COMM_DIALECT_H
#define ENZYMEXLA_COMM_DIALECT_H

#include "mlir/IR/Dialect.h"

#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h.inc"

#endif // ENZYMEXLA_COMM_DIALECT_H
