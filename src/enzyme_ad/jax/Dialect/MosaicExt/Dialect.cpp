#include "Dialect.h"

#include "mlir/IR/Builders.h"

#include "src/enzyme_ad/jax/Dialect/MosaicExt/MosaicExtDialect.cpp.inc"

// Initialize the dialect
void mlir::enzymexla::mosaic_ext::MosaicExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/MosaicExt/MosaicExtOps.cpp.inc"
      >();
}
