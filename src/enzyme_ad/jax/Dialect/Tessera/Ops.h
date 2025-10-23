#ifndef ENZYME_AD_JAX_TESSERA_OPS_H
#define ENZYME_AD_JAX_TESSERA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraOps.h.inc"

#endif // ENZYME_AD_JAX_TESSERA_OPS_H