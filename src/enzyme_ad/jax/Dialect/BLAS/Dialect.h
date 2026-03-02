#ifndef ENZYMEXLA_DIALECT_BLAS_DIALECT_H
#define ENZYMEXLA_DIALECT_BLAS_DIALECT_H

#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/BLAS/BLASDialect.h.inc

#endif
