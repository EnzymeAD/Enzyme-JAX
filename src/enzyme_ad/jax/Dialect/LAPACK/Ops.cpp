#include "Dialect.h"

using namespace mlir;
using namespace mlir::lapack;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/LAPACK/LapackOps.cpp.inc"
