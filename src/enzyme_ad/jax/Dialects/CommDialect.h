#include "mlir/IR/Dialect.h"
#include "mlir/Support/TypeID.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"
#include "src/enzyme_ad/jax/Dialects/CommDialect.h.inc"

#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h" 

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/CommOps.h.inc"