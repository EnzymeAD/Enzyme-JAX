#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_PARAMETERIZEDLOCALIZEDISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

FailureOr<TensorBindingChoice>
buildTensorBindingChoice(ArrayRef<Value> candidateAxes, Value localizedAxis,
                         int64_t tensorAxis) {
  if (!llvm::is_contained(candidateAxes, localizedAxis)) {
    return failure();
  }

  if (tensorAxis < 0) {
    return failure();
  }

  TensorBindingChoice choice;
  choice.localizedAxis = localizedAxis;
  choice.tensorAxis = tensorAxis;
  return choice;
}

FailureOr<Value> findLocalizedAxisForBindings(MeshComputationOp meshOp,
                                             const TensorBindingMap &bindings) {
  SmallVector<Value> logicalAxes;
  logicalAxes.append(meshOp.getSpmdAxes().begin(),
                     meshOp.getSpmdAxes().end());
  logicalAxes.append(meshOp.getMpmdAxes().begin(),
                     meshOp.getMpmdAxes().end());

  for (Value axis : logicalAxes) {
    for (const auto &[tensor, choice] : bindings) {
      (void)tensor;
      if (choice.localizedAxis == axis) {
        return axis;
      }
    }
  }

  return failure();
}

FailureOr<sdy::TensorShardingAttr>
getTensorShardingForMeshValue(MeshComputationOp meshOp, Value tensor) {
  if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
    Region *parentRegion = blockArg.getOwner()->getParent();
    if (!parentRegion || parentRegion->getParentOp() != meshOp.getOperation()) {
      return sdy::getSharding(tensor);
    }

    unsigned argIndex = blockArg.getArgNumber();
    ValueRange inputTensors = meshOp.getInputTensors();
    if (argIndex < inputTensors.size()) {
      return sdy::getSharding(inputTensors[argIndex]);
    }
  }

  return sdy::getSharding(tensor);
}

sdy::TensorShardingAttr
buildIndexLocalizedSharding(sdy::TensorShardingAttr sharding,
                            StringRef localizedAxisName) {
  SmallVector<sdy::DimensionShardingAttr> updatedDimShardings;
  updatedDimShardings.reserve(sharding.getDimShardings().size());
  for (sdy::DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    SmallVector<sdy::AxisRefAttr> keptAxes;
    keptAxes.reserve(dimSharding.getAxes().size());
    for (sdy::AxisRefAttr axisRef : dimSharding.getAxes()) {
      if (axisRef.getName() != localizedAxisName) {
        keptAxes.push_back(axisRef);
      }
    }

    std::optional<int64_t> priority =
        keptAxes.empty() && dimSharding.getIsClosed()
            ? std::nullopt
            : dimSharding.getPriority();
    updatedDimShardings.push_back(sdy::DimensionShardingAttr::get(
        sharding.getContext(), keptAxes, dimSharding.getIsClosed(), priority));
  }

  SmallVector<sdy::AxisRefAttr> updatedReplicatedAxes;
  for (sdy::AxisRefAttr axisRef : sharding.getReplicatedAxes()) {
    if (axisRef.getName() != localizedAxisName) {
      updatedReplicatedAxes.push_back(axisRef);
    }
  }

  SmallVector<sdy::AxisRefAttr> updatedUnreducedAxes;
  for (sdy::AxisRefAttr axisRef : sharding.getUnreducedAxes()) {
    if (axisRef.getName() != localizedAxisName) {
      updatedUnreducedAxes.push_back(axisRef);
    }
  }

  return sdy::TensorShardingAttr::get(
      sharding.getContext(), sharding.getMeshOrRef(), updatedDimShardings,
      updatedReplicatedAxes, updatedUnreducedAxes);
}

void dumpBindingLookup(Value clonedTensor, Value originalTensor,
             int64_t deviceIndex, const TensorBindingChoice &choice,
             bool matched) {
  llvm::errs() << "[localize-distributed] binding lookup: clone="
               << clonedTensor << " original=" << originalTensor
               << " localized-axis=" << choice.localizedAxis
         << " tensor-axis=" << choice.tensorAxis
         << " chosen-device-index=" << choice.chosenDeviceIndex
         << " device-index=" << deviceIndex
         << " matched=" << (matched ? "true" : "false") << '\n';
}

FailureOr<TensorBindingChoice>
getBindingChoiceForClonedOp(const DenseMap<Value, Value> &clonedToOriginal,
                            const TensorBindingMap &bindings,
                            Operation *clonedOp, int64_t deviceIndex) {
  std::optional<TensorBindingChoice> candidateChoice;
  std::optional<Value> candidateOriginal;

  for (Value clonedResult : clonedOp->getResults()) {
    if (!isa<ShapedType>(clonedResult.getType())) {
      continue;
    }

    auto originalIt = clonedToOriginal.find(clonedResult);
    if (originalIt == clonedToOriginal.end()) {
      continue;
    }

    auto bindingIt = bindings.find(originalIt->second);
    if (bindingIt == bindings.end()) {
      continue;
    }

    const TensorBindingChoice &choice = bindingIt->second;
    if (!candidateChoice) {
      candidateChoice = choice;
      candidateOriginal = originalIt->second;
      continue;
    }

    if (candidateChoice->localizedAxis != choice.localizedAxis ||
        candidateChoice->tensorAxis != choice.tensorAxis ||
        candidateChoice->chosenDeviceIndex != choice.chosenDeviceIndex) {
      return failure();
    }
  }

  if (!candidateChoice) {
    return failure();
  }

  llvm::errs() << "[localize-distributed] op binding decision: clone="
               << *clonedOp << " original=" << *candidateOriginal
               << " device-index=" << deviceIndex
               << " chosen-device-index=" << candidateChoice->chosenDeviceIndex
               << '\n';
  return *candidateChoice;
}

LogicalResult eraseClonedOpAndSendUsers(Operation *clonedOp) {
  SmallVector<Operation *> sendUsers;
  for (Value result : clonedOp->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (isa<distributed::SendOp>(user)) {
        sendUsers.push_back(user);
        continue;
      }

      return clonedOp->emitOpError()
             << "expected cloned op users to be sends when removing the wrong device index clone";
    }
  }

  for (Operation *user : sendUsers) {
    user->erase();
  }

  clonedOp->erase();
  return success();
}

LogicalResult applyLocalizedBindingsToClone(
  MeshComputationOp sourceMesh, Value localizedAxis,
  StringRef localizedAxisName, int64_t deviceIndex,
  const TensorBindingMap &bindings,
  const DenseMap<Value, Value> &clonedToOriginal) {
  for (const auto &[clonedTensor, originalTensor] : clonedToOriginal) {
    if (!isa<ShapedType>(clonedTensor.getType())) {
      continue;
    }

    auto bindingIt = bindings.find(originalTensor);
    if (bindingIt == bindings.end()) {
      llvm::errs() << "[localize-distributed] binding lookup: clone="
                   << clonedTensor << " original=" << originalTensor
                   << " -> <no binding>\n";
      continue;
    }

    const TensorBindingChoice &choice = bindingIt->second;
    bool matched = choice.localizedAxis == localizedAxis &&
                   choice.chosenDeviceIndex == deviceIndex;
    dumpBindingLookup(clonedTensor, originalTensor, deviceIndex, choice,
                      matched);

    if (!matched) {
      continue;
    }

    FailureOr<sdy::TensorShardingAttr> originalSharding =
        getTensorShardingForMeshValue(sourceMesh, originalTensor);
    if (failed(originalSharding)) {
      return sourceMesh.emitOpError()
             << "expected bound tensor to have explicit sharding";
    }

    sdy::setSharding(
        clonedTensor,
        buildIndexLocalizedSharding(*originalSharding, localizedAxisName));

    llvm::errs() << "[localize-distributed] applied sharding: clone="
           << clonedTensor << " sharding=" << sdy::getSharding(clonedTensor)
           << '\n';
  }

  return success();
}

LogicalResult cloneMeshComputationForLocalizedAxis(
    MeshComputationOp meshOp, Value localizedAxis,
    const TensorBindingMap &bindings) {
  SmallVector<Value> oldSpmdAxes(meshOp.getSpmdAxes().begin(),
                                 meshOp.getSpmdAxes().end());
  SmallVector<Value> oldMpmdAxes(meshOp.getMpmdAxes().begin(),
                                 meshOp.getMpmdAxes().end());

  bool axisIsSpmd = llvm::is_contained(oldSpmdAxes, localizedAxis);
  bool axisIsMpmd = llvm::is_contained(oldMpmdAxes, localizedAxis);
  if (!axisIsSpmd && !axisIsMpmd) {
    return meshOp.emitOpError()
           << "selected localization axis is not present on the mesh";
  }

  if (!axisIsSpmd) {
    meshOp.emitRemark()
        << "selected axis is already in the MPMD partition; no move required";
    return success();
  }

  auto axisNameOr = meshOp.findShardyAxisNameForLogicalAxis(localizedAxis);
  if (failed(axisNameOr)) {
    return meshOp.emitOpError()
           << "failed to resolve the recorded Shardy axis name for the "
              "localized axis";
  }
  StringRef localizedAxisName = axisNameOr->getValue();

  int64_t axisSize = static_cast<int64_t>(getAxisSize(
      TypedOpResult<LogicalCommAxisType>(localizedAxis)));
  if (axisSize <= 0) {
    return meshOp.emitOpError()
           << "expected localized axis to have a positive size";
  }

  SmallVector<Value> newSpmdAxes;
  newSpmdAxes.reserve(oldSpmdAxes.size());
  for (Value axis : oldSpmdAxes) {
    if (axis != localizedAxis) {
      newSpmdAxes.push_back(axis);
    }
  }

  SmallVector<Value> newMpmdAxes = oldMpmdAxes;
  newMpmdAxes.push_back(localizedAxis);

  SmallVector<Attribute> newShardyAxisNameAttrs;
  newShardyAxisNameAttrs.reserve(newSpmdAxes.size() + newMpmdAxes.size());
  for (Value axis : newSpmdAxes) {
    auto axisNameOr = meshOp.findShardyAxisNameForLogicalAxis(axis);
    if (failed(axisNameOr)) {
      return meshOp.emitOpError()
             << "failed to resolve the Shardy axis name for a retained SPMD "
                "axis";
    }
    newShardyAxisNameAttrs.push_back(*axisNameOr);
  }
  for (Value axis : newMpmdAxes) {
    auto axisNameOr = meshOp.findShardyAxisNameForLogicalAxis(axis);
    if (failed(axisNameOr)) {
      return meshOp.emitOpError()
             << "failed to resolve the Shardy axis name for a retained MPMD "
                "axis";
    }
    newShardyAxisNameAttrs.push_back(*axisNameOr);
  }

  uint32_t oldDeviceBodies = meshOp.getNumDeviceBodies();
  if (oldDeviceBodies == 0) {
    return meshOp.emitOpError() << "expected at least one device body";
  }

  uint32_t newDeviceBodies = oldDeviceBodies * static_cast<uint32_t>(axisSize);
  uint32_t numCommunicationBodies = meshOp.getNumCommunicationBodies();
  uint32_t totalBodies = newDeviceBodies + numCommunicationBodies;

  IRRewriter rewriter(meshOp.getContext());
  rewriter.setInsertionPoint(meshOp);

  auto localizedMesh = rewriter.create<MeshComputationOp>(
      meshOp.getLoc(), meshOp->getResultTypes(), newSpmdAxes, newMpmdAxes,
      rewriter.getArrayAttr(newShardyAxisNameAttrs), meshOp.getInputTensors(),
      newDeviceBodies, numCommunicationBodies, totalBodies);

  for (uint32_t oldDeviceIndex = 0; oldDeviceIndex < oldDeviceBodies;
       ++oldDeviceIndex) {
    Region &sourceRegion = meshOp.getDeviceBody(oldDeviceIndex);
    Block &sourceBlock = sourceRegion.front();
    for (int64_t axisCoord = 0; axisCoord < axisSize; ++axisCoord) {
      uint32_t newDeviceIndex = static_cast<uint32_t>(
          axisCoord * static_cast<int64_t>(oldDeviceBodies) + oldDeviceIndex);
      IRMapping mapper;
      Region &destRegion = localizedMesh.getDeviceBody(newDeviceIndex);
      sourceRegion.cloneInto(&destRegion, mapper);

      DenseMap<Value, Value> clonedToOriginal;
      for (Operation &oldOp : sourceBlock) {
        for (Value oldResult : oldOp.getResults()) {
          if (Value clonedResult = mapper.lookupOrNull(oldResult)) {
            clonedToOriginal[clonedResult] = oldResult;
          }
        }
      }

      if (failed(applyLocalizedBindingsToClone(meshOp, localizedAxis,
                                               localizedAxisName, axisCoord,
                                               bindings,
                                               clonedToOriginal))) {
        return failure();
      }

      SmallVector<Operation *> clonedOps;
      for (Operation &clonedOp : destRegion.front()) {
        clonedOps.push_back(&clonedOp);
      }

      for (Operation *clonedOp : llvm::reverse(clonedOps)) {
        if (!clonedOp || clonedOp->getNumResults() == 0) {
          continue;
        }

        FailureOr<TensorBindingChoice> choiceOr =
            getBindingChoiceForClonedOp(clonedToOriginal, bindings, clonedOp,
                                        axisCoord);
        if (failed(choiceOr)) {
          continue;
        }

        if (choiceOr->chosenDeviceIndex != axisCoord) {
          if (failed(eraseClonedOpAndSendUsers(clonedOp))) {
            return failure();
          }
        }
      }
    }
  }

  SmallVector<Value> oldAxes = oldSpmdAxes;
  oldAxes.append(oldMpmdAxes.begin(), oldMpmdAxes.end());
  for (Value axis : oldAxes) {
    auto oldCommunicationBodyIndexOr =
        meshOp.findCommunicationBodyIndexForAxis(axis);
    auto newCommunicationBodyIndexOr =
        localizedMesh.findCommunicationBodyIndexForAxis(axis);
    if (failed(oldCommunicationBodyIndexOr) ||
        failed(newCommunicationBodyIndexOr)) {
      return meshOp.emitOpError()
             << "failed to map communication bodies while localizing axis";
    }

    IRMapping mapper;
    meshOp.getCommunicationBody(*oldCommunicationBodyIndexOr).cloneInto(
        &localizedMesh.getCommunicationBody(*newCommunicationBodyIndexOr),
        mapper);
  }

  (void)bindings;
  rewriter.replaceOp(meshOp, localizedMesh.getResults());
  return success();
}

LogicalResult parameterizedLocalizeMeshComputation(
    MeshComputationOp meshOp, const TensorBindingMap &bindings) {
  if (bindings.empty()) {
    meshOp.emitError()
        << "parameterized localization requires at least one tensor binding";
    return failure();
  }

  SmallVector<Value> localizationAxes;
  localizationAxes.append(meshOp.getSpmdAxes().begin(),
                          meshOp.getSpmdAxes().end());
  localizationAxes.append(meshOp.getMpmdAxes().begin(),
                          meshOp.getMpmdAxes().end());
  if (localizationAxes.empty()) {
    meshOp.emitError() << "parameterized localization requires mesh axes";
    return failure();
  }

  for (const auto &[tensor, choice] : bindings) {
    if (!isa<ShapedType>(tensor.getType())) {
      meshOp.emitError()
          << "parameterized localization expects tensor-shaped SSA values";
      return failure();
    }
    FailureOr<TensorBindingChoice> validatedChoice =
        buildTensorBindingChoice(localizationAxes, choice.localizedAxis,
                                 choice.tensorAxis);
    if (failed(validatedChoice)) {
      meshOp.emitError()
          << "parameterized localization received a binding for an axis that "
             "is not present on the mesh computation";
      return failure();
    }
  }

  FailureOr<Value> localizedAxisOr =
      findLocalizedAxisForBindings(meshOp, bindings);
  if (failed(localizedAxisOr)) {
    meshOp.emitError()
        << "parameterized localization requires one binding to identify the "
           "axis to move";
    return failure();
  }

  if (failed(cloneMeshComputationForLocalizedAxis(meshOp, *localizedAxisOr,
                                                  bindings))) {
    return failure();
  }
  return success();
}

struct ParameterizedLocalizeDistributedPass
    : public enzyme::distributed::impl::
          ParameterizedLocalizeDistributedPassBase<
              ParameterizedLocalizeDistributedPass> {
  using ParameterizedLocalizeDistributedPassBase::
      ParameterizedLocalizeDistributedPassBase;

  void runOnOperation() override {
    MeshComputationOp meshOp = getOperation();
    SmallVector<Value> localizationAxes;
    localizationAxes.append(meshOp.getSpmdAxes().begin(),
                            meshOp.getSpmdAxes().end());
    localizationAxes.append(meshOp.getMpmdAxes().begin(),
                            meshOp.getMpmdAxes().end());
    if (localizationAxes.empty()) {
      meshOp.emitError() << "parameterized localization requires mesh axes";
      signalPassFailure();
      return;
    }

    if (targetMpmdAxisIndex < 0 ||
        targetMpmdAxisIndex >= static_cast<int64_t>(localizationAxes.size())) {
      meshOp.emitError()
          << "target-mpmd-axis-index must refer to an existing axis";
      signalPassFailure();
      return;
    }

    Value localizedAxis =
        localizationAxes[static_cast<size_t>(targetMpmdAxisIndex)];

    TensorBindingMap bindings;
    for (Value tensor : meshOp.getInputTensors()) {
      if (!isa<ShapedType>(tensor.getType())) {
        continue;
      }

      if (tensorAxis < 0 ||
          tensorAxis >= cast<ShapedType>(tensor.getType()).getRank()) {
        meshOp.emitError()
            << "tensor-axis must be within the rank of every shaped input";
        signalPassFailure();
        return;
      }

      TensorBindingChoice choice;
      choice.localizedAxis = localizedAxis;
      choice.tensorAxis = tensorAxis;
      bindings[tensor] = choice;
    }

    if (failed(parameterizedLocalizeMeshComputation(meshOp, bindings))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir
