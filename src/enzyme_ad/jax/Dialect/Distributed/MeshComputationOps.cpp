#include "CollectiveOps.h"
#include "Dialect.h"

namespace mlir::enzyme::distributed {

LogicalResult MeshComputationOp::verify() {
  Block &bodyBlock = getBody().front();
  for (Operation &op : bodyBlock.getOperations()) {
    bool isMetadataOp = op.hasTrait<OpTrait::enzyme::axis::MetadataTrait>();
    bool isDistributedFunction = isa<DistributedFunctionOp>(op);
    if (!isMetadataOp && !isDistributedFunction) {
      return emitOpError()
             << "only distributed.Function and static metadata ops are "
             << "allowed in the mesh body; operation '" << op.getName()
             << "' is neither";
    }
  }
  return success();
}

} // namespace mlir::enzyme::distributed

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
