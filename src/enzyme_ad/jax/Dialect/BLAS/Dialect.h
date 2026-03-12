#ifndef ENZYME_AD_JAX_DIALECT_BLAS_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_BLAS_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/Base.h"

#include "src/enzyme_ad/jax/Dialect/BLAS/BlasDialect.h.inc"
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasAttrs.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasOps.h.inc"

namespace mlir::blas {
using BlasDialect = BLASDialect;
} // namespace mlir::blas

#endif
