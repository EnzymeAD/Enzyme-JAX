#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyDialect.cpp.inc"

// Initialize the dialect
void mlir::enzyme::perfify::PerfifyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.cpp.inc"
      >();
}
