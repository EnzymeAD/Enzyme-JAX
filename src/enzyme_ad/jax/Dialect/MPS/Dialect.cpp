#include "src/enzyme_ad/jax/Dialect/MPS/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/MPS/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/MPS/MPSDialect.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/MPS/MPSEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/MPS/MPSAttrs.cpp.inc"

void mlir::mps::MPSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/MPS/MPSAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/MPS/MPSOps.cpp.inc"
      >();
}
