#include "src/enzyme_ad/jax/Dialects/CommDialect.h"
#include "src/enzyme_ad/jax/Dialects/CommDialect.cpp.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/CommOps.cpp.inc"

using namespace mlir;
using namespace mlir::comm;

void CommunicationDialect::initialize() {
  addOperations<CommFoo>(); // Register CommFoo operation
  addOperations<CommSplitBranch>();
  addOperations<CommJoin>();
}

// CommunicationDialect::CommunicationDialect(mlir::MLIRContext*){}