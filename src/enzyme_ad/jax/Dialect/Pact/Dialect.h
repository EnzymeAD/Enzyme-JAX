#ifndef ENZYME_AD_JAX_DIALECT_PACT_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_PACT_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/enzyme_ad/jax/Dialect/Pact/PactDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Pact/PactAttrs.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Pact/PactOps.h.inc"

#endif // ENZYME_AD_JAX_DIALECT_PACT_DIALECT_H
