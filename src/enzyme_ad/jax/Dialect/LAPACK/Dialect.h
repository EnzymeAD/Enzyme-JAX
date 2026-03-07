#ifndef ENZYMEXLA_DIALECT_LAPACK_DIALECT_H
#define ENZYMEXLA_DIALECT_LAPACK_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "stablehlo/dialect/Base.h"

#include "src/enzyme_ad/jax/Dialect/BLAS/Dialect.h"

#include "src/enzyme_ad/jax/Dialect/LAPACK/LapackDialect.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/LAPACK/LapackOps.h.inc"

namespace mlir::lapack {
using LapackDialect = LAPACKDialect;
} // namespace mlir::lapack

#endif
