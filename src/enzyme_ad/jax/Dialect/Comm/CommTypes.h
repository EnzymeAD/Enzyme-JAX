#ifndef ENZYME_AD_JAX_DIALECTS_COMM_COMMTYPES_H
#define ENZYME_AD_JAX_DIALECTS_COMM_COMMTYPES_H


#include "mlir/Support/TypeID.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"

#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.h.inc"

#endif