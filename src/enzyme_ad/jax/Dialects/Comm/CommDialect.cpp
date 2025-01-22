#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.cpp.inc"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::comm;

void CommunicationDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialects/Comm/CommTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"
      >();
}


#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommTypes.cpp.inc"