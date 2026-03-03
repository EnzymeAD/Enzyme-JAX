#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/LAPACK/LapackDialect.cpp.inc"

void mlir::enzymexla::lapack::LAPACKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/LAPACK/LapackOps.cpp.inc"
      >();
}
