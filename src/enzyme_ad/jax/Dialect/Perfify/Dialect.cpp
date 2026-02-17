#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyTypes.cpp.inc"

// Initialize the dialect
void mlir::enzyme::perfify::PerfifyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.cpp.inc"
      >();
}