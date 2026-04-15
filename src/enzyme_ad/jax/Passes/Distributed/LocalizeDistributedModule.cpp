#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_LOCALIZEDISTRIBUTEDMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

bool producesTensorValue(Operation &op) {
  return llvm::any_of(op.getResultTypes(), [](Type type) {
    return isa<ShapedType>(type);
  });
}

void recordTensorProducingOps(MeshComputationOp meshOp,
                             SmallVectorImpl<Operation *> &tensorOps) {
  Region &deviceBody = meshOp.getDeviceBody(0);
  if (deviceBody.empty()) {
    return;
  }

  for (Operation &op : deviceBody.front()) {
    if (isa<distributed::RecvOp>(op) || !producesTensorValue(op)) {
      continue;
    }
    tensorOps.push_back(&op);
  }
}

TensorBindingMap buildPrototypeTensorBindingMap(
  ArrayRef<Value> localizationAxes, ArrayRef<Operation *> tensorOps,
  int64_t startAxisIndex, int64_t tensorAxis,
  int64_t chosenDeviceCount) {
  TensorBindingMap bindings;
  if (localizationAxes.empty() || tensorOps.empty()) {
    return bindings;
  }

  for (auto [tensorOpIndex, op] : llvm::enumerate(tensorOps)) {
    Value localizedAxis = localizationAxes[
        (static_cast<int64_t>(tensorOpIndex) + startAxisIndex) %
        static_cast<int64_t>(localizationAxes.size())];
    TensorBindingChoice choice;
    choice.localizedAxis = localizedAxis;
    choice.tensorAxis = tensorAxis;
    choice.chosenDeviceIndex = chosenDeviceCount > 0
                     ? static_cast<int64_t>(tensorOpIndex) %
                       chosenDeviceCount
                     : 0;

    for (Value result : op->getResults()) {
      if (!isa<ShapedType>(result.getType())) {
        continue;
      }
      bindings[result] = choice;
    }
  }

  return bindings;
}

struct LocalizeDistributedModulePass
    : public enzyme::distributed::impl::LocalizeDistributedModulePassBase<
          LocalizeDistributedModulePass> {
  using LocalizeDistributedModulePassBase::LocalizeDistributedModulePassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    llvm::SmallVector<MeshComputationOp> meshComputations;
    moduleOp.walk([&](MeshComputationOp meshOp) { meshComputations.push_back(meshOp); });

    for (auto [meshIndex, meshOp] : llvm::enumerate(meshComputations)) {
      SmallVector<Value> localizationAxes;
      localizationAxes.append(meshOp.getSpmdAxes().begin(),
                              meshOp.getSpmdAxes().end());
      if (localizationAxes.empty()) {
        localizationAxes.append(meshOp.getMpmdAxes().begin(),
                                meshOp.getMpmdAxes().end());
      }

      if (localizationAxes.empty()) {
        meshOp.emitRemark()
            << "skipping localization because the mesh has no axes to cycle";
        continue;
      }

      SmallVector<Operation *> tensorOps;
      recordTensorProducingOps(meshOp, tensorOps);
      if (tensorOps.empty()) {
        meshOp.emitRemark()
            << "no tensor-producing ops found in the computational region";
        continue;
      }

      int64_t targetMpmdAxisIndex =
          static_cast<int64_t>(meshIndex % localizationAxes.size());
      int64_t tensorAxis = 0;
        Value localizedAxis = localizationAxes[static_cast<size_t>(targetMpmdAxisIndex)];
        int64_t axisSize = static_cast<int64_t>(getAxisSize(
          TypedOpResult<LogicalCommAxisType>(localizedAxis)));
        TensorBindingMap bindings = buildPrototypeTensorBindingMap(
          localizationAxes, tensorOps, targetMpmdAxisIndex, tensorAxis,
          axisSize);

      meshOp.emitRemark()
          << "prototype localization config: axis-index="
          << targetMpmdAxisIndex << ", tensor-axis=Index"
          << ", tensor-axis-value=" << tensorAxis
          << ", chosen-device-count=" << axisSize
          << ", cycle-slot=" << meshIndex
          << ", hlo-ops=" << tensorOps.size()
          << ", bindings=" << bindings.size();

      if (failed(parameterizedLocalizeMeshComputation(meshOp, bindings))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir