#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme::perfify;

namespace mlir::enzyme::perfify {} // namespace mlir::enzyme::perfify

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.cpp.inc"
