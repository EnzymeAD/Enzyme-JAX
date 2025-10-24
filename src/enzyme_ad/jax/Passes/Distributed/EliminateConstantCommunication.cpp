/**
 * Replaces send(constant); recv(); with just constant.
 */

#include "Passes.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {
#define GEN_PASS_DEF_ELIMINATECONSTANTCOMMUNICATIONPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

bool isConstantOp(Operation *op) { return isa<stablehlo::ConstantOp>(op); }
bool isConstant(Value val) {
  if (auto op = val.getDefiningOp()) {
    return isConstantOp(op);
  }
  return false;
}

struct EliminateConstantCommunicationPass
    : public impl::EliminateConstantCommunicationPassBase<
          EliminateConstantCommunicationPass> {
  using EliminateConstantCommunicationPassBase::
      EliminateConstantCommunicationPassBase;
  void runOnOperation() override {
    Operation *op = getOperation();
    // Post-order walk is allowed to erase the sends. Less sure if we
    // are permitted to erase the recvs during the walk.
    op->walk([&](enzyme::distributed::SendOp send) {
      if (isConstant(send.getValue())) {
        // Check that we are the only sender on this channel, and get
        // the corresponding recvs.
        auto users = getTokenUsers(send.getToken());
        if (!isSoleSender(send, send.getToken(), users)) {
          // If we're not the sole sender, we can't eliminate the communication.
          return;
        }
        // If we are the sole sender, we can replace all recvs with a copy of
        // the constant value. However, since the recv may be in a different
        // scope, we need to replace it with a clone of the constant op.
        for (auto user : users) {
          if (auto recv = dyn_cast<enzyme::distributed::RecvOp>(user)) {
            auto cloned_const = send.getValue().getDefiningOp()->clone();
            // Insert the cloned constant right before the recv
            recv->getBlock()->getOperations().insert(recv->getIterator(),
                                                     cloned_const);
            recv.getResult().replaceAllUsesWith(cloned_const->getResult(0));
            recv.erase();
          }
        }
        send.erase();
      }
    });
  }
};
} // namespace mlir::enzyme::distributed