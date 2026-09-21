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

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace mlir::enzyme::axis {
void materializeMaybeTemporaryOp(::mlir::Operation *op, ::mlir::Block &block);

// Storage width for every axis/factor/mesh extent (and cumulative offset) in
// this dialect and in Dialect/Distributed. 64 bits because an extent is often
// a product of factors that each fit in 32 bits but whose product does not.
// Declared ahead of the generated type/interface headers so ODS parameters
// can name it.
using AxisExtentT = uint64_t;

// Storage width for every axis-factor stride. A stride is a spacing in the
// same index space an extent measures, so it needs the same width. It is a
// separate alias so the two roles stay distinguishable in signatures.
using AxisStrideT = uint64_t;
} // namespace mlir::enzyme::axis

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Axis/AxisDialect.h.inc"
// Op interfaces
#include "src/enzyme_ad/jax/Dialect/Axis/AxisInterfaces.h.inc"
// Type interfaces
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypeInterfaces.h.inc"
// Types
#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypes.h.inc"
// Ops
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisOps.h.inc"

#endif // ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_H
