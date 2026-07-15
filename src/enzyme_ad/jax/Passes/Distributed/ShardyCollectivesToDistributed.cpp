#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_SHARDYCOLLECTIVESTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

/**
 * Some collectives (the reduces) have computational behavior that isn't
 * fully captured just by the sharding annotations, but most of the other
 * ops are capturable just by the sharding annotations and can be rewritten
 * as a group. This function identifies the latter.
 */
bool sharding_encapsulates_behavior(Operation *op) {
  bool captured = isa<sdy::AllGatherOp>(op) || isa<sdy::AllSliceOp>(op) ||
                  isa<sdy::AllToAllOp>(op) || isa<sdy::CollectivePermuteOp>(op);
  return captured;
}

// TODO: hacky horrible quick prototype! Should use shardy op interfaces /
// getter methods or should record sharding for values before rewriting.
FailureOr<sdy::TensorShardingAttr> getTensorShardingFromValue(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto meshComputation = dyn_cast<MeshComputationOp>(parentOp)) {
      unsigned argIndex = blockArg.getArgNumber();
      auto inputTensors = meshComputation.getInputTensors();
      if (argIndex < inputTensors.size()) {
        return sdy::getSharding(inputTensors[argIndex]);
      }
    }
    return sdy::getSharding(value);
  }

  if (isa<OpResult>(value)) {
    // can now call into the shardy api since we have skipped the
    // mesh region
    return sdy::getSharding(value);
  }

  return failure();
}

FailureOr<llvm::StringMap<Value>>
buildShardyAxisToDistributedAxisMap(MeshComputationOp meshComputation,
                                    SmallVectorImpl<Value> &collectiveAxes) {
  auto shardyAxisNames = meshComputation.getShardyAxisNames();
  if (!shardyAxisNames) {
    return failure();
  }

  SmallVector<Value> distributedAxes;
  distributedAxes.reserve(meshComputation.getSpmdAxes().size() +
                          meshComputation.getMpmdAxes().size());
  for (Value axis : meshComputation.getSpmdAxes()) {
    distributedAxes.push_back(axis);
  }
  for (Value axis : meshComputation.getMpmdAxes()) {
    distributedAxes.push_back(axis);
  }
  collectiveAxes.assign(distributedAxes.begin(), distributedAxes.end());
  if (shardyAxisNames.size() != distributedAxes.size()) {
    return failure();
  }

  llvm::StringMap<Value> axisMap;
  for (Attribute shardyAxisNameAttr : shardyAxisNames) {
    auto shardyAxisName = dyn_cast<StringAttr>(shardyAxisNameAttr);
    if (!shardyAxisName) {
      return failure();
    }
    FailureOr<Value> logicalAxis =
        meshComputation.findLogicalAxisForShardyAxisName(
            shardyAxisName.getValue());
    if (failed(logicalAxis)) {
      return failure();
    }
    axisMap[shardyAxisName.getValue()] = *logicalAxis;
  }

  return axisMap;
}

FailureOr<SmallVector<Value>>
mapShardyAxisRefsToDistributedAxes(ArrayRef<sdy::AxisRefAttr> shardyAxes,
                                   const llvm::StringMap<Value> &axisMap) {
  SmallVector<Value> mappedAxes;
  mappedAxes.reserve(shardyAxes.size());
  for (sdy::AxisRefAttr shardyAxis : shardyAxes) {
    if (shardyAxis.getSubAxisInfo()) {
      return failure();
    }

    auto axisIt = axisMap.find(shardyAxis.getName());
    if (axisIt == axisMap.end()) {
      return failure();
    }
    mappedAxes.push_back(axisIt->second);
  }
  return mappedAxes;
}

FailureOr<SmallVector<Value>>
buildDistributedShardingForTensor(sdy::TensorShardingAttr sharding,
                                  const llvm::StringMap<Value> &axisMap,
                                  IRRewriter &rewriter, Location loc) {
  if (!sharding.getUnreducedAxes().empty()) {
    return failure();
  }

  SmallVector<Value> distributedShardings;
  distributedShardings.reserve(sharding.getDimShardings().size());
  for (sdy::DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    FailureOr<SmallVector<Value>> logicalAxes =
        mapShardyAxisRefsToDistributedAxes(dimSharding.getAxes(), axisMap);
    if (failed(logicalAxes)) {
      return failure();
    }

    auto shardingOp = rewriter.create<ShardingOp>(loc, *logicalAxes);
    distributedShardings.push_back(shardingOp.getSharding());
  }

  return distributedShardings;
}

LogicalResult lowerTensorTransfer(Operation *op, Value tensorIn,
                                  Type tensorOutType,
                                  sdy::TensorShardingAttr inputSharding,
                                  sdy::TensorShardingAttr outputSharding,
                                  IRRewriter &rewriter) {
  auto meshComputation = op->getParentOfType<MeshComputationOp>();
  if (!meshComputation) {
    return failure();
  }

  SmallVector<Value> collectiveAxes;
  FailureOr<llvm::StringMap<Value>> axisMap =
      buildShardyAxisToDistributedAxisMap(meshComputation, collectiveAxes);
  if (failed(axisMap)) {
    op->emitError() << "failed to build Shardy-to-distributed axis mapping";
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(meshComputation);

  FailureOr<SmallVector<Value>> inputShardingAxes =
      buildDistributedShardingForTensor(inputSharding, *axisMap, rewriter,
                                        op->getLoc());
  if (failed(inputShardingAxes)) {
    op->emitError()
        << "failed to lower input Shardy sharding to distributed sharding";
    return failure();
  }

  FailureOr<SmallVector<Value>> outputShardingAxes =
      buildDistributedShardingForTensor(outputSharding, *axisMap, rewriter,
                                        op->getLoc());
  if (failed(outputShardingAxes)) {
    op->emitError()
        << "failed to lower output Shardy sharding to distributed sharding";
    return failure();
  }

  auto globalInputTensorTypeAttr = TypeAttr::get(tensorIn.getType());
  auto globalOutputTensorTypeAttr = TypeAttr::get(tensorOutType);

  // Before localization, shardy-to-distributed keeps tensor SSA values in
  // global shape and uses collective metadata for sharding semantics.
  auto localInputTensorTypeAttr = globalInputTensorTypeAttr;
  auto localOutputTensorTypeAttr = globalOutputTensorTypeAttr;

  auto collective = rewriter.create<CollectiveOp>(
      op->getLoc(), MessageTokenType::get(rewriter.getContext()),
      collectiveAxes, globalInputTensorTypeAttr, globalOutputTensorTypeAttr,
      localInputTensorTypeAttr, localOutputTensorTypeAttr, *inputShardingAxes,
      *outputShardingAxes);

  rewriter.setInsertionPoint(op);
  rewriter.create<SendOp>(op->getLoc(), collective.getToken(), tensorIn);
  auto recv = rewriter.create<RecvOp>(op->getLoc(), tensorOutType,
                                      collective.getToken());
  sdy::setSharding(recv.getMessage(), outputSharding);

  if (meshComputation.getNumCommunicationBodies() == 0) {
    op->emitError()
        << "expected at least one communication region for transfer op";
    return failure();
  }

  Value transferAxis;
  if (!collectiveAxes.empty()) {
    transferAxis = collectiveAxes.front();
  }
  if (!outputShardingAxes->empty()) {
    if (auto shardingOp =
            outputShardingAxes->front().getDefiningOp<ShardingOp>()) {
      if (!shardingOp.getAxes().empty()) {
        transferAxis = shardingOp.getAxes().front();
      }
    }
  }

  unsigned communicationBodyIndex = 0;
  if (transferAxis) {
    auto communicationBodyIndexOr =
        meshComputation.findCommunicationBodyIndexForAxis(transferAxis);
    if (succeeded(communicationBodyIndexOr)) {
      communicationBodyIndex = *communicationBodyIndexOr;
    }
  }

  Region &communicationBody =
      meshComputation.getCommunicationBody(communicationBodyIndex);
  Block &communicationBlock = communicationBody.front();
  auto transferInsertIt = communicationBlock.end();
  if (!communicationBlock.empty()) {
    transferInsertIt = Block::iterator(communicationBlock.getTerminator());
  }
  rewriter.setInsertionPoint(&communicationBlock, transferInsertIt);
  rewriter.create<TransferOp>(op->getLoc(), collective.getToken());

  rewriter.replaceOp(op, recv.getMessage());
  return success();
}

LogicalResult lowerCollective(Operation *op, IRRewriter &rewriter) {
  // Find a shardy collective op in scope, child to a MeshComputationOp.
  auto sdyCollective = dyn_cast<sdy::CollectiveOpInterface>(op);
  if (!sdyCollective) {
    return failure();
  }

  auto meshComputation = op->getParentOfType<MeshComputationOp>();
  if (!meshComputation) {
    return failure();
  }

  if (op->getNumResults() != 1) {
    op->emitError() << "expected single-result Shardy collective op";
    return failure();
  }

  if (!sharding_encapsulates_behavior(op)) {
    return failure();
  }

  Value tensorIn = sdyCollective.getTensor();
  Type tensorOutType = op->getResult(0).getType();

  FailureOr<sdy::TensorShardingAttr> inputSharding =
      getTensorShardingFromValue(tensorIn);
  if (failed(inputSharding)) {
    op->emitError() << "expected input tensor to have explicit Shardy sharding";
    return failure();
  }

  sdy::TensorShardingAttr outputSharding = sdyCollective.getOutSharding();
  return lowerTensorTransfer(op, tensorIn, tensorOutType, *inputSharding,
                             outputSharding, rewriter);
}

LogicalResult lowerReshard(sdy::ReshardOp reshard, IRRewriter &rewriter) {
  auto meshComputation = reshard->getParentOfType<MeshComputationOp>();
  if (!meshComputation) {
    return failure();
  }

  Value tensorIn = reshard.getInput();
  Type tensorOutType = reshard.getResult().getType();

  FailureOr<sdy::TensorShardingAttr> inputSharding =
      getTensorShardingFromValue(tensorIn);
  if (failed(inputSharding)) {
    reshard.emitError()
        << "expected reshard input tensor to have explicit sharding";
    return failure();
  }

  return lowerTensorTransfer(reshard, tensorIn, tensorOutType, *inputSharding,
                             reshard.getSharding(), rewriter);
}

struct ShardyCollectivesToDistributedPass
    : public enzyme::distributed::impl::ShardyCollectivesToDistributedPassBase<
          ShardyCollectivesToDistributedPass> {
  using ShardyCollectivesToDistributedPassBase::
      ShardyCollectivesToDistributedPassBase;

  void runOnOperation() override {
    if (failed(rewriteShardyCollectivesInFunction(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

llvm::LogicalResult rewriteShardyCollectivesInFunction(func::FuncOp funcOp) {
  // Use a deterministic source-order walk rather than greedy pattern rewrite,
  // so send/recv/transfer lowering preserves collective order.
  SmallVector<Operation *> collectivesInOrder;
  funcOp.walk([&](Operation *op) {
    if ((isa<sdy::CollectiveOpInterface>(op) &&
         sharding_encapsulates_behavior(op)) ||
        isa<sdy::ReshardOp>(op)) {
      collectivesInOrder.push_back(op);
    }
  });

  IRRewriter rewriter(funcOp.getContext());
  for (Operation *collective : collectivesInOrder) {
    if (!collective || !collective->getBlock()) {
      continue;
    }
    rewriter.setInsertionPoint(collective);
    if (auto reshard = dyn_cast<sdy::ReshardOp>(collective)) {
      if (failed(lowerReshard(reshard, rewriter))) {
        return failure();
      }
      continue;
    }

    if (failed(lowerCollective(collective, rewriter))) {
      return failure();
    }
  }

  return success();
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir
