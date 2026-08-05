#include "Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

// Include the .cpp.inc files
#include "src/enzyme_ad/jax/Dialect/Axis/AxisDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypes.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypeInterfaces.cpp.inc"

void mlir::enzyme::axis::AxisDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Axis/AxisOps.cpp.inc"
      >();
}
