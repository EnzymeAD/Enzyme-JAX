#include "Ops.h"
#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::enzymexla::comm;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/MPIOps.cpp.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/NCCLOps.cpp.inc"
