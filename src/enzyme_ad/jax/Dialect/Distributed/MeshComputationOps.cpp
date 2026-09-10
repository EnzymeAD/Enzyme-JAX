#include "Dialect.h"

namespace mlir::enzyme::distributed {

LogicalResult MeshComputationOp::verify() {
  Block &bodyBlock = getBody().front();
  for (Operation &op : bodyBlock.getOperations()) {
    if (!op.hasTrait<OpTrait::enzyme::axis::MetadataTrait>()) {
      return emitOpError()
             << "only static metadata ops are allowed in the mesh body; "
             << "operation '" << op.getName() << "' is not marked with "
             << "metadata trait";
    }
  }
  return success();
}

} // namespace mlir::enzyme::distributed

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
