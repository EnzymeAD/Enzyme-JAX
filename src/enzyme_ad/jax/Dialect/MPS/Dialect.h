#ifndef ENZYMEXLA_MPS_DIALECT_H
#define ENZYMEXLA_MPS_DIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include "src/enzyme_ad/jax/Dialect/MPS/MPSDialect.h.inc"

#include "src/enzyme_ad/jax/Dialect/MPS/MPSEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/MPS/MPSAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES

#endif // ENZYMEXLA_MPS_DIALECT_H
