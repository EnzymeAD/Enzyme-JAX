#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/BLAS/BlasDialect.cpp.inc"
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasAttrs.cpp.inc"

void mlir::blas::BLASDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasOps.cpp.inc"
      >();
}
