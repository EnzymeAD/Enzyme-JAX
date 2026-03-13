#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Pact/PactAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Pact/PactOps.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Pact/PactDialect.cpp.inc"

void mlir::enzyme::pact::PactDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Pact/PactAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Pact/PactOps.cpp.inc"
      >();
}
