#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"

using namespace mlir;
using namespace mlir::comm;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"