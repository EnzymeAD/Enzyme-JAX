#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Dialect.h"

namespace mlir::enzyme::distributed {

LogicalResult RegionComputationOp::verify() {
  // TODO
  return mlir::success();
}
} // namespace mlir::enzyme::distributed
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"