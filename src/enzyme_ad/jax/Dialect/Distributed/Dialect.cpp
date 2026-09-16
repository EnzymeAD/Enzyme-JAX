#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"

// Include the .cpp.inc files
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedInterfaces.cpp.inc"

// Initialize the dialect
void mlir::enzyme::distributed::DistributedDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
      >();
}