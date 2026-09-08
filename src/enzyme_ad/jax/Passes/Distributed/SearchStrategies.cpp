#include "src/enzyme_ad/jax/Passes/Distributed/BeamSearchDriver.h"
#include "src/enzyme_ad/jax/Passes/Distributed/LogicalAxisOverlap.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ReplayTree.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include <memory>
#include <utility>

namespace mlir::enzyme::distributed {
using namespace mlir::enzyme::axis;

#define GEN_PASS_DEF_DISTRIBUTEDSEARCHSTRATEGIESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

/**
 * Plan for this pass:
 * To discover efficient distribution plans, search over mappings from logical
 * axes to either physical device axes, temporal pipelines (TODO), or
 * within-device axes (i.e. serialization, though we don't care if the device
 * chooses to internally parallelize.) We use beam search for this:
 *  - Ahead of search, freeze an order of axes to decide. This deduplicates
 *    search effort, since different decision orders lead to the same result.
 *    This should attempt to place "expensive" / impactful axes earlier in the
 *    order heuristically.
 *  - For an axis being decided, make decisions for subfactors at a time rather
 *    than the whole axis at once, allowing hybrid strategies. Exception:
 *    serialization decisions should take the entire remaining axis at once.
 *  - Decide factors from outermost / least contiguous to innermost. This means
 *    that we serialize local chunks of tensors. (Not a hard invariant: reshapes
 *    can introduce strides)
 *  - Currently doing fixed budget beam, may change in the future
 *  - TODO: figure out how we can avoid invoking the device cost model for
 *    identical / unchanged kernels. Should only need to recalculate kernels
 *    and collectives affected by the latest decision.
 *  - To avoid needing to mantain multiple modules (including IR mapping
 *    bookeeping) we maintain a single module and replay decisions onto it.
 *    during each lowering pass.
 */
namespace {

template <typename OpTy>
using SharedOpRef = std::shared_ptr<mlir::OwningOpRef<OpTy>>;

// Converts a typed value to a SharedOpRef by extracting its defining op
template <typename ValT, typename OpT>
inline SharedOpRef<OpT> sharedOpRefFromValue(ValT value) {
  if (auto defOp = value.template getDefiningOp<OpT>()) {
    auto owningRef = mlir::OwningOpRef<OpT>(defOp);
    return std::make_shared<decltype(owningRef)>(std::move(owningRef));
  }
  return nullptr;
}

// Converts a SharedOpRef to its typed result value
template <typename ValT, typename OpT>
inline ValT sharedOpRefToValue(const SharedOpRef<OpT> &opRef,
                               unsigned resultIndex) {
  assert(opRef && *opRef && "Invalid shared op reference");
  Value result = (*opRef)->getResult(resultIndex);
  return cast<ValT>(result);
}

// Converts a SharedOpRef to its typed result value for ops with a single unique
// result (e.g., AxisFactorOp which has a typed no-argument getResult())
template <typename ValT, typename OpT>
inline ValT sharedOpRefToUniqueValue(const SharedOpRef<OpT> &opRef) {
  assert(opRef && *opRef && "Invalid shared op reference");
  Value result = (*opRef)->getResult();
  return cast<ValT>(result);
}

// Lifts unique ownership for transformed space values that came from owned inputs.
// Pass-through values reuse their existing ownership; new values get fresh ownership.
// This prevents duplicate ownership of the same underlying op.
template <typename ValT, typename OpT>
std::vector<SharedOpRef<OpT>>
liftUniqueOwnership(llvm::ArrayRef<ValT> transformedSpace,
                    llvm::ArrayRef<SharedOpRef<OpT>> ownedInputs) {
  // Build a map from input values to their owning refs
  llvm::DenseMap<Value, SharedOpRef<OpT>> inputValueToRef;
  for (const auto &opRef : ownedInputs) {
    if (opRef) {
      Value val = sharedOpRefToUniqueValue<ValT, OpT>(opRef);
      inputValueToRef[val] = opRef;
    }
  }

  std::vector<SharedOpRef<OpT>> result;
  for (ValT resultValue : transformedSpace) {
    auto it = inputValueToRef.find(resultValue);
    if (it != inputValueToRef.end()) {
      // This value came from an input - reuse its ownership
      result.push_back(it->second);
    } else {
      // This is a new/transformed value - create fresh ownership
      auto opRef = sharedOpRefFromValue<ValT, OpT>(cast<ValT>(resultValue));
      if (opRef) {
        result.push_back(opRef);
      }
    }
  }
  return result;
}

// Collects all logical axes with users in the given module.
static std::vector<TypedValue<LogicalMeshAxisType>>
findAllLogicalAxes(ModuleOp moduleOp) {
  std::vector<TypedValue<LogicalMeshAxisType>> logicalAxes;
  moduleOp.walk([&](LogicalMeshAxesOp axesOp) {
    // Check dead axes have been cleaned up so we don't have to search over them
    for (Value axis : axesOp.getAxes()) {
      bool hasUsers = !axis.use_empty();
#ifndef NDEBUG
      static bool remarked = false;
      if (!remarked && !hasUsers) {
        axesOp.emitWarning("Skipping dead logical axis: will unnecessarily "
                           "increase search space. Skipping future remarks.");
        remarked = true;
      }
#endif
      if (hasUsers)
        logicalAxes.push_back(cast<TypedValue<LogicalMeshAxisType>>(axis));
    }
  });
  return logicalAxes;
}

static llvm::SmallVector<SharedOpRef<AxisFactorOp>>
findAllPhysicalAxes(ModuleOp moduleOp, mlir::OpBuilder &builder,
                    Location loc) {
  llvm::SmallVector<SharedOpRef<AxisFactorOp>> physicalFactors;
  GetPhysicalMeshAxesOp getAxesOp = nullptr;
  int count = 0;
  moduleOp.walk([&](GetPhysicalMeshAxesOp op) {
    getAxesOp = op;
    count++;
  });

  assert(getAxesOp && count == 1 &&
         "Expected exactly one GetPhysicalMeshAxesOp");

  auto factorValues = viewAxesAsFactors(getAxesOp.getAxes(), builder, loc);
  for (TypedValue<AxisFactorType> axisValue : factorValues) {
    auto opRef = sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(
        axisValue);
    if (opRef) {
      physicalFactors.push_back(opRef);
    }
  }
  return physicalFactors;
}

using LogicalAxisOrder = std::vector<TypedValue<LogicalMeshAxisType>>;

struct AxisMappingReplayState {
  llvm::DenseMap<TypedValue<LogicalMeshAxisType>,
                 std::vector<SharedOpRef<AxisFactorOp>>>
      axisMapping;
  void apply(const AxisMappingReplayState &other) {
    for (auto &entry : other.axisMapping) {
      auto &vec = axisMapping[entry.first];
      vec.insert(vec.end(), entry.second.begin(), entry.second.end());
    }
  }
};

class AxisMappingReplayTree
    : public mlir::enzyme::distributed::ReplayTree<AxisMappingReplayTree,
                                                   AxisMappingReplayState> {
public:
  using mlir::enzyme::distributed::ReplayTree<
      AxisMappingReplayTree, AxisMappingReplayState>::ReplayTree;

  // Returns what, if any, factors are currently bound to
  // the given logical axis in the current replay tree.
  llvm::SmallVector<SharedOpRef<AxisFactorOp>>
  getBindings(TypedValue<LogicalMeshAxisType> axis) {
    struct LookupOperator {
      ReplayQueryShortCircuit
      operator()(const AxisMappingReplayState &state,
                 TypedValue<LogicalMeshAxisType> axis,
                 llvm::SmallVector<SharedOpRef<AxisFactorOp>> &result) {
        auto it = state.axisMapping.find(axis);
        if (it != state.axisMapping.end()) {
          result.insert(result.end(), it->second.rbegin(), it->second.rend());
        }
        return mlir::enzyme::distributed::Continue;
      }
    };
    LookupOperator lookup;
    llvm::SmallVector<SharedOpRef<AxisFactorOp>> result;
    queryReplayReverse(lookup, axis, result);
    std::reverse(result.begin(), result.end());
    return result;
  }

  llvm::SmallVector<SharedOpRef<AxisFactorOp>>
  lookupBindings(llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> axes) {
    llvm::SmallVector<SharedOpRef<AxisFactorOp>> result;
    for (auto axis : axes) {
      auto bindings = getBindings(axis);
      result.insert(result.end(), bindings.begin(), bindings.end());
    }
    return result;
  }
};

class StrategySearchNode : public BeamSearchNodeBase {

  std::shared_ptr<const LogicalAxisOrder> axes;
  std::size_t axisIndex;
  std::shared_ptr<AxisMappingReplayTree> decisions;
  int extentTaken; // extent already taken from the current axis

  // Factors of the physical mesh available for this axis
  std::vector<SharedOpRef<AxisFactorOp>> availableSpace;

  StrategySearchNode() = delete; // disable default constructor

  StrategySearchNode(const StrategySearchNode &other,
                     std::shared_ptr<AxisMappingReplayTree> childDecisions)
      : axes(other.axes), axisIndex(other.axisIndex), decisions(childDecisions),
        extentTaken(other.extentTaken), availableSpace(other.availableSpace) {}

public:
  StrategySearchNode(std::shared_ptr<const LogicalAxisOrder> axes,
                     std::shared_ptr<AxisMappingReplayTree> decisions)
      : axes(axes), axisIndex(-1 /* incremented to 0 upon setupNextAxis */),
        decisions(decisions), extentTaken(1) {
    assert(extentTaken >= 1);
  }

  TypedValue<LogicalMeshAxisType> currentAxis() const {
    assert(!finalized());
    return (*axes)[axisIndex];
  }
  bool finalized() const override { return axisIndex >= axes->size(); }

  std::shared_ptr<StrategySearchNode> makeChild() const {
    auto childDecisions = AxisMappingReplayTree::makeChild(decisions);
    auto childNode = std::shared_ptr<StrategySearchNode>(
        new StrategySearchNode(*this, childDecisions));
    return childNode;
  }

  void setupNextAxis(LogicalAxisOverlap &overlap,
                     llvm::ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace,
                     OpBuilder &builder) {
    extentTaken = 1;
    axisIndex++;
    if (finalized()) {
      return;
    }
    availableSpace = std::vector<SharedOpRef<AxisFactorOp>>(
        totalMeshSpace.begin(), totalMeshSpace.end());

    auto axis = currentAxis();

    // Look up what decisions have already been made for any overlapping axes,
    // then remove those from availableSpace.
    auto overlappingAxes = overlap.getOverlaps(axis);
    if (!overlappingAxes) {
      return;
    }
    auto overlappingBinds = decisions->lookupBindings(*overlappingAxes);

    // Extract result values from the factor ops to use with existing utilities
    // The SharedOpRef ensures the ops survive this scope, so we can safely
    // use non-owning references to their result values
    llvm::SmallVector<TypedValue<AxisFactorType>> availableSpaceValues;
    for (const auto &factorOpRef : availableSpace) {
      availableSpaceValues.push_back(
          sharedOpRefToUniqueValue<TypedValue<AxisFactorType>, AxisFactorOp>(
              factorOpRef));
    }

    llvm::SmallVector<TypedValue<AxisFactorType>> overlappingBindsValues;
    for (const auto &factorOpRef : overlappingBinds) {
      overlappingBindsValues.push_back(
          sharedOpRefToUniqueValue<TypedValue<AxisFactorType>, AxisFactorOp>(
              factorOpRef));
    }

    // Use axis dialect utilities to get the space subtraction
    auto remainingSpace =
        subtractSpace(availableSpaceValues, overlappingBindsValues, builder);
    if (failed(remainingSpace)) {
      availableSpace.clear();
      return;
    }

    // Lift unique ownership: pass-through values reuse their original refs,
    // while new/transformed values get fresh ownership
    availableSpace =
        liftUniqueOwnership<TypedValue<AxisFactorType>, AxisFactorOp>(
            *remainingSpace, availableSpace);
  }
};

class StrategyExplorer : public BeamSearchExplorerBase<StrategySearchNode> {
public:
  StrategyExplorer() : BeamSearchExplorerBase<StrategySearchNode>() {}

  virtual std::vector<std::shared_ptr<StrategySearchNode>>
  generateCandidatesFromNode(
      std::shared_ptr<StrategySearchNode> node) override {
    // TODO Implement the exploration logic here.
    return {};
  }
};

class StrategyScorer : public BeamSearchScorerBase<StrategySearchNode> {
public:
  virtual double
  score(const std::shared_ptr<StrategySearchNode> &node) override {
    // TODO Implement the scoring logic here.
    return 0.0;
  }
};

struct DistributedSearchStrategiesPass
    : public impl::DistributedSearchStrategiesPassBase<
          DistributedSearchStrategiesPass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    mlir::OpBuilder builder(moduleOp.getContext());
    builder.clearInsertionPoint();

    auto overlap = LogicalAxisOverlap(moduleOp);

    auto logicalAxes = findAllLogicalAxes(moduleOp);
    llvm::SmallVector<SharedOpRef<AxisFactorOp>> physicalAxes =
      findAllPhysicalAxes(moduleOp, builder, moduleOp.getLoc());

    // TODO: order by importance
    auto axes =
        std::make_shared<const LogicalAxisOrder>(std::move(logicalAxes));
    auto decisions = AxisMappingReplayTree::makeRoot();

    auto initialNode = std::make_shared<StrategySearchNode>(axes, decisions);
    initialNode->setupNextAxis(overlap, physicalAxes, builder);

    int TODO_PARAMETER_BEAM_SIZE = 100;
    BeamSearchBreadthFirstQueue<StrategySearchNode> queue(
        TODO_PARAMETER_BEAM_SIZE);
    queue.push(initialNode);
    StrategyExplorer explorer;
    StrategyScorer scorer;

    BeamSearchDriver<StrategySearchNode> driver(queue, scorer, explorer);
    driver.run();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
