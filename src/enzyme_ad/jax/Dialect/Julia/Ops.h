#ifndef ENZYME_AD_JAX_DIALECT_JULIA_OPS_H
#define ENZYME_AD_JAX_DIALECT_JULIA_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaOps.h.inc"
#undef GET_OP_CLASSES

#endif // ENZYME_AD_JAX_DIALECT_JULIA_OPS_H
