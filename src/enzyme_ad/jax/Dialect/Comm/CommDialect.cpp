#include "src/enzyme_ad/jax/Dialect/Comm/Comm.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::comm;

#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.cpp.inc"

void CommDialect::initialize() {
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.cpp.inc"
  >();

  addOperations<
    #define GET_OP_LIST
    #include "src/enzyme_ad/jax/Dialect/Comm/CommOps.cpp.inc"
  >();
}
