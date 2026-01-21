#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraAttrs.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraDialect.cpp.inc"

// Initialize the dialect
void mlir::enzyme::tessera::TesseraDialect::initialize() {
addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraAttrs.cpp.inc"
>();

addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraOps.cpp.inc"
      >();
}
