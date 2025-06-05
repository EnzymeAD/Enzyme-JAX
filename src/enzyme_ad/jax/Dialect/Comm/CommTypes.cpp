#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.h"


#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.cpp.inc"