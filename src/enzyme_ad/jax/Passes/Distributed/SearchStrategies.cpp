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

static std::vector<TypedValue<AxisFactorType>>
findAllPhysicalAxes(ModuleOp moduleOp) {
  assert(false && "findAllPhysicalAxes not implemented");
  return {};
}

using LogicalAxisOrder = std::vector<TypedValue<LogicalMeshAxisType>>;

struct AxisMappingReplayState {
  llvm::DenseMap<TypedValue<LogicalMeshAxisType>,
                 std::vector<TypedValue<AxisFactorType>>>
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
  llvm::SmallVector<TypedValue<AxisFactorType>>
  getBindings(TypedValue<LogicalMeshAxisType> axis) {
    struct LookupOperator {
      ReplayQueryShortCircuit
      operator()(const AxisMappingReplayState &state,
                 TypedValue<LogicalMeshAxisType> axis,
                 llvm::SmallVector<TypedValue<AxisFactorType>> &result) {
        auto it = state.axisMapping.find(axis);
        if (it != state.axisMapping.end()) {
          result.insert(result.end(), it->second.rbegin(), it->second.rend());
        }
        return mlir::enzyme::distributed::Continue;
      }
    };
    LookupOperator lookup;
    llvm::SmallVector<TypedValue<AxisFactorType>> result;
    queryReplayReverse(lookup, axis, result);
    std::reverse(result.begin(), result.end());
    return result;
  }

  llvm::SmallVector<TypedValue<AxisFactorType>>
  lookupBindings(llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> axes) {
    llvm::SmallVector<TypedValue<AxisFactorType>> result;
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
  std::vector<TypedValue<AxisFactorType>> availableSpace;

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
                     std::vector<TypedValue<AxisFactorType>> totalMeshSpace,
                     OpBuilder &builder) {
    extentTaken = 1;
    axisIndex++;
    if (finalized()) {
      return;
    }
    availableSpace = std::move(totalMeshSpace);

    auto axis = currentAxis();

    // Look up what decisions have already been made for any overlapping axes,
    // then remove those from availableSpace.
    auto overlappingAxes = overlap.getOverlaps(axis);
    if (!overlappingAxes) {
      return;
    }
    auto overlappingBinds = decisions->lookupBindings(*overlappingAxes);

    // Use axis dialect utilities to get the space subtraction
    // of binds from availableSpace.
    // TODO: memory leak?
    auto remainingSpace =
        subtractSpace(availableSpace, overlappingBinds, builder);
    if (failed(remainingSpace)) {
      availableSpace.clear();
      return;
    }
    availableSpace.assign(remainingSpace->begin(), remainingSpace->end());
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
    mlir::OpBuilder temporaryBuilder(moduleOp.getContext());
    temporaryBuilder.clearInsertionPoint();

    auto overlap = LogicalAxisOverlap(moduleOp);

    auto logicalAxes = findAllLogicalAxes(moduleOp);
    std::vector<TypedValue<AxisFactorType>> physicalAxes =
        findAllPhysicalAxes(moduleOp);
    // TODO: order by importance
    auto axes =
        std::make_shared<const LogicalAxisOrder>(std::move(logicalAxes));
    auto decisions = AxisMappingReplayTree::makeRoot();

    auto initialNode = std::make_shared<StrategySearchNode>(axes, decisions);
    initialNode->setupNextAxis(overlap, physicalAxes, temporaryBuilder);

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
