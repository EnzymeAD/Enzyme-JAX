#include "CollectiveOps.h"
#include "Dialect.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

LogicalResult MeshComputationOp::verify() {
  Block &bodyBlock = getBody().front();
  for (Operation &op : bodyBlock.getOperations()) {
    bool isMetadataOp = op.hasTrait<OpTrait::enzyme::axis::MetadataTrait>();
    bool isDistributedFunction = isa<DistributedFunctionOp>(op);
    bool isStablehloConstant = isa<stablehlo::ConstantOp>(op);
    if (!isMetadataOp && !isDistributedFunction && !isStablehloConstant) {
      return emitOpError()
             << "only distributed.Function, stablehlo.constant, and static "
                "metadata ops are "
             << "allowed in the mesh body; operation '" << op.getName()
             << "' is neither";
    }
  }
  return success();
}

} // namespace mlir::enzyme::distributed

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
