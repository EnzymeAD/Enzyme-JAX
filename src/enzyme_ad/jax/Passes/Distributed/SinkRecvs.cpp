#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_DISTRIBUTEDSINKRECVSPASS
#define GEN_PASS_DEF_DISTRIBUTEDSINKRECVSMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

/// Sink all RecvOps in `block` to just before their earliest consumer.
/// Within a single-block region this is safe as long as the recv stays after
/// its defining CollectiveOp (which we guarantee because the CollectiveOp
/// precedes the recv in SSA order and we only move ops downward).
void sinkRecvsInBlock(Block &block) {
  // Collect recvs in a stable snapshot so we can freely mutate the block.
  SmallVector<RecvOp> recvs;
  for (Operation &op : block)
    if (auto recv = dyn_cast<RecvOp>(&op))
      recvs.push_back(recv);

  for (RecvOp recv : recvs) {
    Value msg = recv.getMessage();
    if (msg.use_empty())
      continue;

    // Find the earliest user by scanning the block forward.
    Operation *firstUser = nullptr;
    for (Operation &op : block) {
      if (op.isBeforeInBlock(recv.getOperation()))
        continue; // op is above the recv — skip (shouldn't hold users)
      if (&op == recv.getOperation())
        continue;
      for (Value operand : op.getOperands()) {
        if (operand == msg) {
          if (!firstUser || op.isBeforeInBlock(firstUser))
            firstUser = &op;
          break;
        }
      }
    }

    if (!firstUser)
      continue;

    // Move recv to just before its first user (no-op if already there).
    recv->moveBefore(firstUser);
  }
}

/// Apply recv sinking to every device body of `meshOp`.
void sinkRecvs(MeshComputationOp meshOp) {
  for (uint32_t i = 0; i < meshOp.getNumDeviceBodies(); ++i) {
    Region &body = meshOp.getDeviceBody(i);
    if (body.empty())
      continue;
    sinkRecvsInBlock(body.front());
  }
}

namespace {
struct DistributedSinkRecvsPass
    : public enzyme::distributed::impl::DistributedSinkRecvsPassBase<
          DistributedSinkRecvsPass> {
  using DistributedSinkRecvsPassBase::DistributedSinkRecvsPassBase;

  void runOnOperation() override { sinkRecvs(getOperation()); }
};

struct DistributedSinkRecvsModulePass
    : public enzyme::distributed::impl::DistributedSinkRecvsModulePassBase<
          DistributedSinkRecvsModulePass> {
  using DistributedSinkRecvsModulePassBase::DistributedSinkRecvsModulePassBase;

  void runOnOperation() override {
    SmallVector<MeshComputationOp> meshOps;
    getOperation().walk([&](MeshComputationOp op) { meshOps.push_back(op); });
    for (MeshComputationOp op : meshOps)
      sinkRecvs(op);
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir
