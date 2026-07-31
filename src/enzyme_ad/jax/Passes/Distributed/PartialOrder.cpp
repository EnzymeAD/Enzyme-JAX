#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"

namespace mlir::enzyme::distributed {

void SSABlockPartialOrderAnalysis::buildPartialOrder() {
  for (Operation &op : block->getOperations()) {
    for (Value result : op.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user->getBlock() != block) {
          continue;
        }
        partial_order.addOrder(&op, user);
      }
    }
  }
  partial_order.close();
}
} // namespace mlir::enzyme::distributed