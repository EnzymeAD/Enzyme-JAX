#include "Dialect.h"

#include "mlir/IR/Builders.h"

#include "src/enzyme_ad/jax/Dialect/TritonExt/TritonExtDialect.cpp.inc"

// Initialize the dialect
void mlir::enzyme::triton_ext::TritonExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/TritonExt/TritonExtOps.cpp.inc"
      >();
}
