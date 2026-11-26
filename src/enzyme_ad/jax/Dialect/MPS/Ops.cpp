#include "src/enzyme_ad/jax/Dialect/MPS/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::mps;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/MPS/MPSOps.cpp.inc"

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }
