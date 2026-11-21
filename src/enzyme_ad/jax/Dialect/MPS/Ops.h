#ifndef ENZYMEXLA_MPS_OPS_H
#define ENZYMEXLA_MPS_OPS_H

#include "src/enzyme_ad/jax/Dialect/MPS/Dialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/MPS/MPSOps.h.inc"
#undef GET_OP_CLASSES

#endif // ENZYMEXLA_MPS_OPS_H
