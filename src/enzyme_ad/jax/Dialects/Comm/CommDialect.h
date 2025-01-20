#ifndef ENZYME_AD_JAX_DIALECTS_COMM_COMMDIALECT_H
#define ENZYME_AD_JAX_DIALECTS_COMM_COMMDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/TypeID.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"
#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h.inc"

#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.h.inc"

#endif