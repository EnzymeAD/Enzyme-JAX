#include "src/enzyme_ad/jax/Passes/Distributed/BeamSearchDriver.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ReplayTree.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include <memory>

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

// Attempt to hybridize replaying decision lists (O(n) per step, so O(n^2))
// vs storing complete state at leaves (O(n) size per leaf, O(n) time to copy)
// to a hybrid approach that ammortizes both to a (hopefully) more efficient
// strategy.

struct LogicalAxisWorklist {
  // SSA value of the next logical axis to rewrite
  TypedValue<LogicalMeshAxisType> axis;
  std::shared_ptr<LogicalAxisWorklist> next;

  static std::shared_ptr<LogicalAxisWorklist>
  fromArray(const llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> &axes) {
    std::shared_ptr<LogicalAxisWorklist> current = nullptr;
    for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
      auto node = std::make_shared<LogicalAxisWorklist>();
      node->axis = *it;
      node->next = current;
      current = node;
    }
    return current;
  }
};

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

  llvm::SmallVector<TypedValue<AxisFactorType>>
  getBindings(TypedValue<LogicalMeshAxisType> axis) {
    struct LookupOperator {
      ReplayQueryShortCircuit
      operator()(const AxisMappingReplayState &state,
                 TypedValue<LogicalMeshAxisType> axis,
                 llvm::SmallVector<TypedValue<AxisFactorType>> &result) {
        auto it = state.axisMapping.find(axis);
        if (it != state.axisMapping.end()) {
          result.insert(result.end(), it->second.begin(), it->second.end());
        }
        return mlir::enzyme::distributed::Continue;
      }
    };
    LookupOperator lookup;
    llvm::SmallVector<TypedValue<AxisFactorType>> result;
    queryReplay(lookup, axis, result);
    return result;
  }
};

struct StrategySearchNode : public BeamSearchNodeBase {

  std::shared_ptr<LogicalAxisWorklist> worklist;
  std::shared_ptr<AxisMappingReplayTree> decisions;
  int extentTaken; // extent already taken from the current axis

  StrategySearchNode() = delete; // disable default constructor
  StrategySearchNode(std::shared_ptr<LogicalAxisWorklist> worklist,
                     std::shared_ptr<AxisMappingReplayTree> decisions,
                     int extentTaken)
      : worklist(worklist), decisions(decisions), extentTaken(extentTaken) {
    assert(extentTaken >= 1);
  }

  TypedValue<LogicalMeshAxisType> currentAxis() const { return worklist->axis; }
  bool finalized() const override { return worklist == nullptr; }
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

    auto logicalAxes = findAllLogicalAxes(moduleOp);
    // TODO: order by importance
    auto worklist = LogicalAxisWorklist::fromArray(logicalAxes);
    auto decisions = AxisMappingReplayTree::makeRoot();

    auto initialNode =
        std::make_shared<StrategySearchNode>(worklist, decisions, 1);

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
