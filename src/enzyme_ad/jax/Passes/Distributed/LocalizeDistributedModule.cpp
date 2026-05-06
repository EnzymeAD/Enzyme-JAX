#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include <limits>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_LOCALIZEDISTRIBUTEDMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

// Forward declarations from other distributed pass TUs.
void runOverlapCommunication(MeshComputationOp meshOp, int64_t kMaxVal);
void sinkRecvs(MeshComputationOp meshOp);

namespace {

bool producesTensorValue(Operation &op) {
  return llvm::any_of(op.getResultTypes(),
                      [](Type type) { return isa<ShapedType>(type); });
}

void recordTensorProducingOps(MeshComputationOp meshOp,
                              SmallVectorImpl<Operation *> &tensorOps) {
  Region &deviceBody = meshOp.getDeviceBody(0);
  if (deviceBody.empty())
    return;
  for (Operation &op : deviceBody.front()) {
    if (isa<distributed::RecvOp>(op) || !producesTensorValue(op))
      continue;
    tensorOps.push_back(&op);
  }
}

// ===----------------------------------------------------------------------===
// Binding candidate generation.
//
// For each tensor-producing op, we enumerate:
//   - Replicated on each axis
//   - Sharded on each axis, for tensor dim 0
//   - IndexBased on each axis, for each device index
// ===----------------------------------------------------------------------===

struct BindingCandidate {
  ShardingMode mode;
  Value axis;
  int64_t tensorAxis;
  int64_t deviceIndex; // only relevant for IndexBased
};

SmallVector<BindingCandidate> generateCandidates(ArrayRef<Value> axes) {
  SmallVector<BindingCandidate> candidates;
  for (Value axis : axes) {
    int64_t axisSize = static_cast<int64_t>(
        getAxisSize(TypedOpResult<LogicalCommAxisType>(axis)));

    // Replicated
    candidates.push_back(
        {ShardingMode::Replicated, axis, /*tensorAxis=*/0, /*deviceIndex=*/0});

    // Sharded along tensor dim 0
    candidates.push_back(
        {ShardingMode::Sharded, axis, /*tensorAxis=*/0, /*deviceIndex=*/0});

    // IndexBased for each device index
    for (int64_t d = 0; d < axisSize; ++d) {
      candidates.push_back({ShardingMode::IndexBased, axis, /*tensorAxis=*/0,
                            /*deviceIndex=*/d});
    }
  }
  return candidates;
}

// ===----------------------------------------------------------------------===
// Build a TensorBindingMap from a vector of per-op candidate binding choices.
// ===----------------------------------------------------------------------===

TensorBindingMap buildBindingMap(ArrayRef<Operation *> tensorOps,
                                 ArrayRef<BindingCandidate> choices) {
  assert(tensorOps.size() == choices.size());
  TensorBindingMap bindings;
  for (auto [op, cand] : llvm::zip(tensorOps, choices)) {
    TensorBindingChoice choice;
    choice.localizedAxis = cand.axis;
    choice.tensorAxis = cand.tensorAxis;
    choice.chosenDeviceIndex = cand.deviceIndex;
    choice.shardingMode = cand.mode;
    for (Value result : op->getResults()) {
      if (isa<ShapedType>(result.getType()))
        bindings[result] = choice;
    }
  }
  return bindings;
}

// ===----------------------------------------------------------------------===
// Evaluate a binding configuration by applying it to a clone of the function,
// running the downstream pass pipeline, and measuring the critical path.
// Returns the critical path time, or a large value on failure.
// ===----------------------------------------------------------------------===

// Forward declarations — implemented as member functions of
// LocalizeDistributedModulePass to access the protected runPipeline.

struct LocalizeDistributedModulePass
    : public enzyme::distributed::impl::LocalizeDistributedModulePassBase<
          LocalizeDistributedModulePass> {
  using LocalizeDistributedModulePassBase::LocalizeDistributedModulePassBase;

  double evaluateBindingConfig(func::FuncOp funcOp,
                               MeshComputationOp origMeshOp,
                               ArrayRef<Operation *> tensorOps,
                               ArrayRef<BindingCandidate> choices) {
    // Deep-clone the enclosing function and temporarily insert into the module
    // so that symbol lookups (meshes, axes, etc.) work during localization.
    auto clonedFunc = cast<func::FuncOp>(funcOp->clone());
    // Give the clone a unique name to avoid symbol conflicts.
    clonedFunc.setName((funcOp.getName() + "__localize_search_clone").str());
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    moduleOp.push_back(clonedFunc);

    // Find the MeshComputationOp in the clone.
    MeshComputationOp clonedMeshOp = nullptr;
    clonedFunc.walk([&](MeshComputationOp op) { clonedMeshOp = op; });
    if (!clonedMeshOp) {
      clonedFunc->erase();
      return std::numeric_limits<double>::max();
    }

    // Build axis mapping: original axis Value → cloned axis Value.
    // Axes appear in the same positional order (SPMD then MPMD).
    DenseMap<Value, Value> axisMap;
    {
      auto origSpmd = origMeshOp.getSpmdAxes();
      auto clonedSpmd = clonedMeshOp.getSpmdAxes();
      for (auto [orig, cloned] : llvm::zip(origSpmd, clonedSpmd))
        axisMap[orig] = cloned;
      auto origMpmd = origMeshOp.getMpmdAxes();
      auto clonedMpmd = clonedMeshOp.getMpmdAxes();
      for (auto [orig, cloned] : llvm::zip(origMpmd, clonedMpmd))
        axisMap[orig] = cloned;
    }

    // Remap the candidates' axis values to the cloned mesh.
    SmallVector<BindingCandidate> remappedChoices(choices.begin(),
                                                  choices.end());
    for (auto &cand : remappedChoices) {
      auto it = axisMap.find(cand.axis);
      if (it != axisMap.end())
        cand.axis = it->second;
    }

    // Build the binding map for the clone.
    // The cloned ops are in the same position in device body 0.
    SmallVector<Operation *> clonedTensorOps;
    recordTensorProducingOps(clonedMeshOp, clonedTensorOps);
    if (clonedTensorOps.size() != tensorOps.size()) {
      clonedFunc->erase();
      return std::numeric_limits<double>::max();
    }

    TensorBindingMap bindings =
        buildBindingMap(clonedTensorOps, remappedChoices);

    // Apply localization.
    if (failed(parameterizedLocalizeMeshComputation(clonedMeshOp, bindings))) {
      clonedFunc->erase();
      return std::numeric_limits<double>::max();
    }

    // Run CSE + simplify-collectives on the clone via pipeline.
    {
      OpPassManager funcPM(func::FuncOp::getOperationName());
      funcPM.addPass(createCSEPass());
      funcPM.addPass(createDistributedSimplifyCollectivesPass());
      if (failed(runPipeline(funcPM, clonedFunc))) {
        clonedFunc->erase();
        return std::numeric_limits<double>::max();
      }
    }

    // Run overlap communication on each MeshComputationOp in the clone.
    clonedFunc.walk([&](MeshComputationOp op) {
      runOverlapCommunication(op, /*kMaxVal=*/4);
    });

    // Sink recvs to just before their first use so consumers can start
    // as soon as their tile arrives rather than waiting behind earlier recvs.
    clonedFunc.walk([&](MeshComputationOp op) { sinkRecvs(op); });

    // Measure the critical path across all MeshComputationOps.
    double maxTime = 0.0;
    clonedFunc.walk([&](MeshComputationOp op) {
      HappensBeforeAnalysis hb(op);
      AffineTimingCostModel costModel;
      TimingAnalysis timing(hb, costModel);
      for (Operation *root : hb.classesInTopologicalOrder()) {
        auto timeRange = timing.getTimeRange(root);
        maxTime = std::max(maxTime, timeRange.second);
      }
    });

    clonedFunc->erase();
    return maxTime;
  }

  // ===--------------------------------------------------------------------===
  // Exhaustive enumeration over the Cartesian product of per-op candidates.
  //
  // Enumerates all combinations one at a time. Each clone is erased before the
  // next is created so at most one duplicate exists at any point.
  // ===--------------------------------------------------------------------===

  SmallVector<BindingCandidate>
  searchBestBindings(func::FuncOp funcOp, MeshComputationOp meshOp,
                     ArrayRef<Operation *> tensorOps,
                     ArrayRef<Value> localizationAxes) {
    auto perOpCandidates = generateCandidates(localizationAxes);
    if (perOpCandidates.empty() || tensorOps.empty())
      return {};

    size_t numOps = tensorOps.size();
    size_t numCandidatesPerOp = perOpCandidates.size();

    // Total configurations = numCandidatesPerOp ^ numOps.
    // Represent the current config as a mixed-radix counter.
    SmallVector<size_t> indices(numOps, 0);

    double bestTime = std::numeric_limits<double>::max();
    SmallVector<BindingCandidate> bestChoices;

    // Enumerate via mixed-radix increment.
    for (;;) {
      // Build the current candidate vector from the indices.
      SmallVector<BindingCandidate> current(numOps);
      for (size_t i = 0; i < numOps; ++i)
        current[i] = perOpCandidates[indices[i]];

      // TODO for testing purposes
      // Require at least one tensor to be placed at an indexed position.
      bool hasIndexBased = llvm::any_of(current, [](const BindingCandidate &c) {
        return c.mode == ShardingMode::IndexBased;
      });
      if (!hasIndexBased) {
        // Increment mixed-radix counter.
        bool done = true;
        for (size_t i = numOps; i-- > 0;) {
          ++indices[i];
          if (indices[i] < numCandidatesPerOp) {
            done = false;
            break;
          }
          indices[i] = 0;
        }
        if (done)
          break;
        continue;
      }

      double time = evaluateBindingConfig(funcOp, meshOp, tensorOps, current);
      bool newBest = time < bestTime;

      if (newBest) {
        bestTime = time;
        bestChoices = current;
      }

      // Increment the mixed-radix counter (rightmost digit first).
      bool done = true;
      for (size_t i = numOps; i-- > 0;) {
        ++indices[i];
        if (indices[i] < numCandidatesPerOp) {
          done = false;
          break;
        }
        indices[i] = 0;
      }
      if (done)
        break;
    }

    return bestChoices;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool localizedAnyMesh = false;
    SmallVector<MeshComputationOp> meshComputations;
    moduleOp.walk(
        [&](MeshComputationOp meshOp) { meshComputations.push_back(meshOp); });

    for (auto [meshIndex, meshOp] : llvm::enumerate(meshComputations)) {
      SmallVector<Value> localizationAxes;
      localizationAxes.append(meshOp.getSpmdAxes().begin(),
                              meshOp.getSpmdAxes().end());
      if (localizationAxes.empty()) {
        localizationAxes.append(meshOp.getMpmdAxes().begin(),
                                meshOp.getMpmdAxes().end());
      }
      if (localizationAxes.empty())
        continue;

      SmallVector<Operation *> tensorOps;
      recordTensorProducingOps(meshOp, tensorOps);
      if (tensorOps.empty())
        continue;

      // Find enclosing function for clone-and-measure.
      auto funcOp = meshOp->getParentOfType<func::FuncOp>();
      if (!funcOp)
        continue;

      // Search for best bindings via exhaustive enumeration.
      SmallVector<BindingCandidate> bestChoices =
          searchBestBindings(funcOp, meshOp, tensorOps, localizationAxes);

      if (bestChoices.empty()) {
        signalPassFailure();
        return;
      }

      // Apply the best configuration to the real IR.
      TensorBindingMap bindings = buildBindingMap(tensorOps, bestChoices);
      if (failed(parameterizedLocalizeMeshComputation(meshOp, bindings))) {
        signalPassFailure();
        return;
      }
      localizedAnyMesh = true;
    }

    (void)localizedAnyMesh;
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir