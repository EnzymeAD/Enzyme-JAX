#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.cpp.inc"

using namespace mlir;
using namespace mlir::comm;


void CommunicationDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"
  >();
  addAttributes<
    #define GET_ATTR_LIST
    #include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.cpp.inc"
  >();
  // TODO types when we need them
}
