#ifndef ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "Traits.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Axis/AxisDialect.h.inc"
// Type interfaces
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypeInterfaces.h.inc"
// Types
#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypes.h.inc"
// Ops
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisOps.h.inc"

#endif // ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_H
