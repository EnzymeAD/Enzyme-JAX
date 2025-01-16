#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "llvm/ADT/TypeSwitch.h"


using namespace mlir;
using namespace mlir::comm;

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.cpp.inc"