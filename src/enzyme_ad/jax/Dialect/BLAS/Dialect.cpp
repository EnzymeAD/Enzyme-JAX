#include "Dialect.h"

#include "mlir/IR/Builders.h"

#include "src/enzyme_ad/jax/Dialect/BLAS/BLASDialect.cpp.inc"

void mlir::enzymexla::blas::BLASDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/BLAS/BLASAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/BLAS/BLASOps.cpp.inc"
      >();
}
