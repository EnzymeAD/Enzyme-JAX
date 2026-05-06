#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_DISTRIBUTEDSIMPLIFYCOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

/// Returns true if the Collective has identical input/output shardings and
/// tensor types (the resharding pattern is a no-op).
bool hasIdentityShardingPattern(CollectiveOp collectiveOp) {
  if (collectiveOp.getGlobalInputTensorType() !=
      collectiveOp.getGlobalOutputTensorType())
    return false;
  if (collectiveOp.getLocalInputTensorType() !=
      collectiveOp.getLocalOutputTensorType())
    return false;

  auto inputSharding = collectiveOp.getInputSharding();
  auto outputSharding = collectiveOp.getOutputSharding();
  if (inputSharding.size() != outputSharding.size())
    return false;
  for (size_t i = 0; i < inputSharding.size(); ++i) {
    if (inputSharding[i] != outputSharding[i])
      return false;
  }
  return true;
}

/// Check whether, in every device body, a token has both a Send and a Recv
/// in the same body (meaning data stays on-device and is not cross-lane).
bool isLocalRoundTrip(Value token, MeshComputationOp meshOp) {
  uint32_t numDeviceBodies = meshOp.getNumDeviceBodies();
  for (uint32_t i = 0; i < numDeviceBodies; ++i) {
    Region &body = meshOp.getDeviceBody(i);
    bool hasSend = false;
    bool hasRecv = false;
    body.walk([&](Operation *op) {
      if (auto send = dyn_cast<SendOp>(op)) {
        if (send.getToken() == token)
          hasSend = true;
      } else if (auto recv = dyn_cast<RecvOp>(op)) {
        if (recv.getToken() == token)
          hasRecv = true;
      }
    });
    // If a body has Recv but no Send (or vice versa), data crosses lanes.
    if (hasSend != hasRecv)
      return false;
  }
  return true;
}

void simplifyCollectivesInFunction(func::FuncOp funcOp) {
  SmallVector<MeshComputationOp> meshOps;
  funcOp.walk([&](MeshComputationOp op) { meshOps.push_back(op); });

  for (MeshComputationOp meshOp : meshOps) {
    // Collect identity CollectiveOps. They are defined in the parent
    // function body (before the MeshComputation), not inside its regions.
    // Find them by scanning the token users of the MeshComputation.
    SmallVector<CollectiveOp> toRemove;

    // Walk the function body for CollectiveOps whose tokens are used by
    // Send/Recv/Transfer inside this meshOp.
    funcOp.walk([&](CollectiveOp collectiveOp) {
      if (!hasIdentityShardingPattern(collectiveOp))
        return;

      Value token = collectiveOp.getToken();

      // Verify the token is actually used by this meshOp's regions.
      bool usedByThisMesh = false;
      for (Operation *user : token.getUsers()) {
        if (meshOp->isAncestor(user)) {
          usedByThisMesh = true;
          break;
        }
      }
      if (!usedByThisMesh)
        return;

      // Check that every device body with a Recv also has a Send
      // (and vice versa) — meaning data stays on the same device.
      if (!isLocalRoundTrip(token, meshOp))
        return;

      toRemove.push_back(collectiveOp);
    });

    if (toRemove.empty())
      continue;

    IRRewriter rewriter(funcOp.getContext());

    for (CollectiveOp collectiveOp : toRemove) {
      Value token = collectiveOp.getToken();

      // In each device body, replace Recv results with the Send value,
      // then erase both.
      uint32_t numDeviceBodies = meshOp.getNumDeviceBodies();
      for (uint32_t i = 0; i < numDeviceBodies; ++i) {
        Region &body = meshOp.getDeviceBody(i);
        SendOp foundSend = nullptr;
        RecvOp foundRecv = nullptr;
        body.walk([&](Operation *op) {
          if (auto send = dyn_cast<SendOp>(op)) {
            if (send.getToken() == token)
              foundSend = send;
          } else if (auto recv = dyn_cast<RecvOp>(op)) {
            if (recv.getToken() == token)
              foundRecv = recv;
          }
        });

        if (foundSend && foundRecv) {
          foundRecv.getResult().replaceAllUsesWith(foundSend.getMessage());
          rewriter.eraseOp(foundRecv);
          rewriter.eraseOp(foundSend);
        }
      }

      // Erase Transfer ops in communication bodies.
      uint32_t numCommBodies = meshOp.getNumCommunicationBodies();
      for (uint32_t i = 0; i < numCommBodies; ++i) {
        Region &commBody = meshOp.getCommunicationBody(i);
        SmallVector<TransferOp> transfers;
        commBody.walk([&](TransferOp transferOp) {
          if (transferOp.getToken() == token)
            transfers.push_back(transferOp);
        });
        for (TransferOp t : transfers)
          rewriter.eraseOp(t);
      }

      // Now safe to erase the Collective itself (no remaining users).
      rewriter.eraseOp(collectiveOp);
    }
  }
}

struct DistributedSimplifyCollectivesPass
    : public enzyme::distributed::impl::DistributedSimplifyCollectivesPassBase<
          DistributedSimplifyCollectivesPass> {
  using DistributedSimplifyCollectivesPassBase::
      DistributedSimplifyCollectivesPassBase;

  void runOnOperation() override {
    simplifyCollectivesInFunction(getOperation());
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir
