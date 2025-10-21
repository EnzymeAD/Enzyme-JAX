#include "Dialect.h"
#include "Ops.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraDialect.cpp.inc"

// Initialize the dialect
void mlir::enzyme::tessera::TesseraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraOps.cpp.inc"
      >();
}
