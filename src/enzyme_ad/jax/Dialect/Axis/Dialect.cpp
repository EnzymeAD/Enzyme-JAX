#include "Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

// Include the .cpp.inc files
#include "src/enzyme_ad/jax/Dialect/Axis/AxisDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypes.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Axis/AxisInterfaces.cpp.inc"
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypeInterfaces.cpp.inc"

// Materializes one maybe-temporary op into the target graph (non-SSACFG!)
// block, then recursively materializes maybe-temporary SSA dependencies.
void mlir::enzyme::axis::materializeMaybeTemporaryOp(::mlir::Operation *op,
                                                     ::mlir::Block &block) {
  if (!op) {
    return;
  }

  if (auto maybeTemporary = dyn_cast<MaybeTemporaryInterface>(op);
      maybeTemporary && maybeTemporary.isManifested()) {
    return;
  }

  if (op->getBlock()) {
    op->moveBefore(&block, block.end());
  } else {
    block.getOperations().push_back(op);
  }

  for (mlir::Value operand : op->getOperands()) {
    mlir::Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      continue;
    }
    if (auto maybeTemporary = dyn_cast<MaybeTemporaryInterface>(definingOp);
        maybeTemporary && !maybeTemporary.isManifested()) {
      maybeTemporary.materialize(block);
    }
  }
}

void mlir::enzyme::axis::AxisDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Axis/AxisTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Axis/AxisOps.cpp.inc"
      >();
}
