#include "Dialect.h"

#include "mlir/IR/Builders.h"

#include "src/enzyme_ad/jax/Dialect/CudaTileExt/CudaTileExtDialect.cpp.inc"

// Initialize the dialect
void mlir::enzymexla::cuda_tile_ext::CudaTileExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/CudaTileExt/CudaTileExtOps.cpp.inc"
      >();
}
