#ifndef ENZYME_AD_JAX_DIALECT_TESSERA_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_TESSERA_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraDialect.h.inc"

// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraOps.h.inc"

#endif // ENZYME_AD_JAX_DIALECT_TESSERA_DIALECT_H