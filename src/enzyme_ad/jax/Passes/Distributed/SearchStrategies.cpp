#include "src/enzyme_ad/jax/Passes/Distributed/BeamSearchDriver.h"
#include "src/enzyme_ad/jax/Passes/Distributed/LogicalAxisOverlap.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ReplayTree.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/MeshCostMetadata.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleGraph.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleSimulator.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
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

/**
 * Organization and lifetime model:
 *
 * This pass searches over many possible ways to bind logical mesh axes to
 * physical/device-local execution resources. During that search we manipulate
 * temporary MLIR metadata ops, such as axis factors and device-local axes, so
 * that we can reuse the existing axis dialect utilities for factor arithmetic.
 * These temporary ops are not part of the input module.
 *
 * Search state is split into two pieces:
 *  - The replay tree stores committed distribution decisions. A decision says
 *    which factor values have been bound to one logical axis. If a decision
 *    creates temporary MLIR metadata, the replay-tree delta for that decision
 *    must keep the temporary ops alive.
 *  - A candidate search node stores the available physical space for the axis
 *    currently being explored. This is candidate-local working state, not a
 *    decision that should be replayed. Available-space factors do not introduce
 *    extra axis declarations, so retaining the factor op itself is enough for
 *    their lifetime.
 *
 * For ordinary search logic, prefer non-owning typed SSA values such as
 * TypedValue<AxisFactorType>. SharedOpRef is only an ownership handle for
 * detached temporary ops that must outlive the helper that created them.
 * Detached ops are built with an OpBuilder that has no insertion point, so MLIR
 * will not own them through a parent block. Be careful about making multiple
 * copies of the owner handle: cloning a SharedOpRef is fine, but spinning up a
 * new one creates two independent owners that will result in double deletion /
 * read-after-free.
 */

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

// Lifts unique ownership for transformed space values that came from owned
// inputs. Pass-through values reuse their existing ownership; new values get
// fresh ownership. This prevents duplicate ownership of the same underlying op.
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
    Value axis = axesOp.getAxis();
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
  });
  return logicalAxes;
}

static llvm::SmallVector<SharedOpRef<AxisFactorOp>>
findAllPhysicalAxes(ModuleOp moduleOp, mlir::OpBuilder &builder, Location loc) {
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
                 std::vector<TypedValue<AxisFactorType>>>
      axisMapping;
  // Store ownership of temporary IR items (axes) supporting the state
  std::vector<SharedOpRef<Operation *>> supportingIR;
  // Retains ownership of each decided factor's underlying (possibly
  // detached) op. axisMapping only stores the non-owning Value, so without
  // this the op -- built via a detached builder during search -- would be
  // destroyed as soon as the caller's own SharedOpRef goes out of scope,
  // leaving a dangling Value for anything that reads the decision back later
  // (e.g. ApplyPartialDecisions).
  std::vector<SharedOpRef<AxisFactorOp>> ownedFactors;

  void apply(const AxisMappingReplayState &other) {
    for (auto &entry : other.axisMapping) {
      auto &vec = axisMapping[entry.first];
      vec.insert(vec.end(), entry.second.begin(), entry.second.end());
    }
    supportingIR.insert(supportingIR.end(), other.supportingIR.begin(),
                        other.supportingIR.end());
    ownedFactors.insert(ownedFactors.end(), other.ownedFactors.begin(),
                        other.ownedFactors.end());
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
  getBindings(llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> axes) {
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
  int64_t extentRemaining;

  // Factors of the physical mesh available for this axis
  std::vector<SharedOpRef<AxisFactorOp>> availableSpace;

  // Identity of this node in the search tree, so a later reader can
  // reconstruct the tree purely from these ids: `uid` names this node,
  // `parentUid` names the node it branched from (absent for the root).
  // Mirrors AxisSymbol's atomic-counter pattern (ShardyLogicalAxisAnalysis.h).
  uint64_t uid;
  std::optional<uint64_t> parentUid;
  inline static std::atomic<uint64_t> nextUid{1};

  StrategySearchNode() = delete; // disable default constructor

  // Picks between makeChild()'s real search-tree branch (fresh uid, this
  // node recorded as parent) and clone()'s identity-preserving working copy
  // (uid/parentUid copied verbatim, nothing newly minted).
  enum class IdentityMode { Branch, Preserve };

  StrategySearchNode(const StrategySearchNode &other,
                     std::shared_ptr<AxisMappingReplayTree> childDecisions,
                     IdentityMode mode)
      : axes(other.axes), axisIndex(other.axisIndex), decisions(childDecisions),
        extentRemaining(other.extentRemaining),
        availableSpace(other.availableSpace),
        uid(mode == IdentityMode::Branch
                ? nextUid.fetch_add(1, std::memory_order_relaxed)
                : other.uid),
        parentUid(mode == IdentityMode::Branch
                      ? std::optional<uint64_t>(other.uid)
                      : other.parentUid) {}

public:
  StrategySearchNode(std::shared_ptr<const LogicalAxisOrder> axes,
                     std::shared_ptr<AxisMappingReplayTree> decisions)
      : axes(axes), axisIndex(-1 /* incremented to 0 upon setupNextAxis */),
        decisions(decisions), availableSpace(), extentRemaining(-1),
        uid(nextUid.fetch_add(1, std::memory_order_relaxed)) {}

  TypedValue<LogicalMeshAxisType> currentAxis() const {
    assert(!finalized());
    return (*axes)[axisIndex];
  }
  bool finalized() const override { return axisIndex >= axes->size(); }
  // How many axes this node has advanced past -- equals axes->size() once
  // finalized(). Search-progress reporting's notion of "depth".
  std::size_t getAxisIndex() const { return axisIndex; }
  std::size_t getTotalAxisCount() const { return axes->size(); }
  int64_t getExtentRemaining() const { return extentRemaining; }
  const auto &getAvailableSpace() const { return availableSpace; }
  uint64_t getUid() const { return uid; }
  std::optional<uint64_t> getParentUid() const { return parentUid; }

  // Branches the search tree: used by StrategyExplorer to turn one node into
  // several real candidates. The child gets a fresh uid with this node as
  // its parent.
  std::shared_ptr<StrategySearchNode> makeChild() const {
    auto childDecisions = AxisMappingReplayTree::makeChild(decisions);
    return std::shared_ptr<StrategySearchNode>(
        new StrategySearchNode(*this, childDecisions, IdentityMode::Branch));
  }

  // Forks a working copy that keeps this node's uid/parentUid. For a caller
  // that needs a derived node only to mutate and discard (e.g. heuristic
  // completion ahead of scoring), so it does not count as a new branch in the
  // search tree.
  std::shared_ptr<StrategySearchNode> clone() const {
    auto childDecisions = AxisMappingReplayTree::makeChild(decisions);
    return std::shared_ptr<StrategySearchNode>(
        new StrategySearchNode(*this, childDecisions, IdentityMode::Preserve));
  }

  void setupNextAxis(LogicalAxisOverlap &overlap,
                     llvm::ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace,
                     OpBuilder &builder) {

    axisIndex++;
    if (finalized()) {
      return;
    }
    availableSpace = std::vector<SharedOpRef<AxisFactorOp>>(
        totalMeshSpace.begin(), totalMeshSpace.end());

    auto axis = currentAxis();
    extentRemaining = getAxisExtent(axis);

    // Look up what decisions have already been made for any overlapping axes,
    // then remove those from availableSpace.
    auto overlappingAxes = overlap.getOverlaps(axis);
    if (!overlappingAxes) {
      return;
    }
    auto overlappingBinds = decisions->getBindings(*overlappingAxes);

    // Extract result values from the factor ops to use with existing utilities
    // The SharedOpRef ensures the ops survive this scope, so we can safely
    // use non-owning references to their result values
    llvm::SmallVector<TypedValue<AxisFactorType>> availableSpaceValues;
    for (const auto &factorOpRef : availableSpace) {
      availableSpaceValues.push_back(
          sharedOpRefToUniqueValue<TypedValue<AxisFactorType>, AxisFactorOp>(
              factorOpRef));
    }

    // Use axis dialect utilities to get the space subtraction
    auto remainingSpace =
        subtractSpace(availableSpaceValues, overlappingBinds, builder);
    assert(succeeded(remainingSpace) && "Failed to subtract space");

    // Lift unique ownership: pass-through values reuse their original refs,
    // while new/transformed values get fresh ownership
    availableSpace =
        liftUniqueOwnership<TypedValue<AxisFactorType>, AxisFactorOp>(
            *remainingSpace, availableSpace);
  }

  // Moves to next axis if this one is complete
  void
  considerNextAxis(LogicalAxisOverlap &overlap,
                   llvm::ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace,
                   OpBuilder &builder) {
    if (extentRemaining == 1) {
      setupNextAxis(overlap, totalMeshSpace, builder);
    }
  }

  void addFactor(SharedOpRef<AxisFactorOp> factor) {
    TypedValue<AxisFactorType> factorValue =
        sharedOpRefToUniqueValue<TypedValue<AxisFactorType>, AxisFactorOp>(
            factor);
    int64_t factorExtent = getFactorExtent(factorValue);
    assert(factorExtent > 1 && "Expect search to make progress");
    assert(extentRemaining >= factorExtent &&
           "Factor exceeds remaining extent");

    // add factor to the taken list for working axis
    decisions->getDelta().axisMapping[currentAxis()].push_back(factorValue);
    // Retain ownership alongside the non-owning Value recorded above.
    decisions->getDelta().ownedFactors.push_back(factor);
    extentRemaining /= factorExtent;
  }

  void addSupportingIR(SharedOpRef<Operation *> op) {
    decisions->getDelta().supportingIR.push_back(op);
  }

  // We expect exact quality between the axis factor op
  // being removed and one in the current space. Does not
  // do space subtraction from partial overlaps.
  void removeAvailableSpace(SharedOpRef<AxisFactorOp> factor) {
    auto it = std::find(availableSpace.begin(), availableSpace.end(), factor);
    if (it != availableSpace.end()) {
      availableSpace.erase(it);
    } else {
      llvm::errs()
          << "Attempted to remove a factor that was not in available space\n";
      assert(false &&
             "Attempted to remove a factor that was not in available space");
    }
  }

  void addAvailableSpace(SharedOpRef<AxisFactorOp> factor) {
    availableSpace.push_back(factor);
  }

  // Returns, for every axis touched so far (everything before axisIndex, plus
  // axisIndex itself if not finalized), the factors committed to it. The
  // in-progress axis (axisIndex, if not finalized) may come back with an
  // empty factor list if it was just advanced to via setupNextAxis and no
  // addFactor call has happened yet.
  llvm::SmallVector<std::pair<TypedValue<LogicalMeshAxisType>,
                              llvm::SmallVector<TypedValue<AxisFactorType>>>>
  getPartialDecisions() const {
    std::size_t count = finalized() ? axes->size() : axisIndex + 1;
    llvm::SmallVector<std::pair<TypedValue<LogicalMeshAxisType>,
                                llvm::SmallVector<TypedValue<AxisFactorType>>>>
        result;
    for (std::size_t i = 0; i < count; ++i) {
      TypedValue<LogicalMeshAxisType> axis = (*axes)[i];
      result.emplace_back(axis, decisions->getBindings(axis));
    }
    return result;
  }
};

/**
 * Sub-pass for the evaluation pipeline. Applies the decisions
 * from a provided search node by rewriting the axis.
 *
 * The decisions reference axis Values from the module the search actually
 * ran over. This pass can either apply them to that same module directly
 * (`createInPlace`, `originalToCloned == nullptr`, decided axis Values are
 * used as-is) or replay them onto a clone of it (`create`, used for the
 * disposable scoring/dump clones in `cloneAndApplyDecisions`), in which case
 * an IR mapper is required to go from the original to the clone's axes. We
 * also expect each logical axis to currently not be factored: any factors
 * should have the whole extent of the axis. This pass effectively replaces
 * any products of the axis factors with ones using the decisions from the
 * search node.
 *
 * (Note: we will want this replacement logic elsewhere too, so ideally we
 * implement it as a rewrite in `dialect/axis/Utilities.h` with a general
 * signature)
 *
 * (Note: we should check for users that aren't products, and may have
 * to redesign if we find any that aren't products.)
 */
class ApplyPartialDecisions
    : public PassWrapper<ApplyPartialDecisions, OperationPass<ModuleOp>> {
private:
  std::shared_ptr<StrategySearchNode> node;
  IRMapping *originalToCloned;
  // Records, for in-place application only, splices of detached
  // search-created ops (see resolve() below) keyed by their original Value,
  // so a second decision referencing the same op reuses the splice instead
  // of duplicating it. Serves the same role originalToCloned serves for the
  // disposable-clone case.
  IRMapping inPlaceSplices;

  ApplyPartialDecisions(std::shared_ptr<StrategySearchNode> node,
                        IRMapping *originalToCloned)
      : node(node), originalToCloned(originalToCloned) {}

  // Resolves a Value from the module the decisions were made against to its
  // counterpart in the module this pass is actually rewriting.
  //
  // Most decided Values (logical mesh axes, physical mesh factors) are
  // already real ops belonging to that source module. Applying in place,
  // those resolve to themselves; replaying onto a disposable clone, they
  // resolve through the IR mapper (populated by the clone that created the
  // whole module).
  //
  // A factor's provenance axis (getFactorProvenanceAxis) can instead be a
  // detached, search-created op with no counterpart in either target, such as
  // a device-local serialization axis from applySerializeRemaining that only
  // the search tree's shared_ptrs keep alive. It is cloned into the target
  // module the first time it is needed. The clone recurses over the op's
  // operands, although a provenance op is always a single hop: every
  // AxisFactorOp's axis operand resolves directly to its root axis (see the
  // TemporaryOpGuard comment in Dialect/Axis/Utilities.h).
  Value resolve(Value original, OpBuilder &builder) {
    Operation *defOp = original.getDefiningOp();
    if (!defOp || defOp->getBlock())
      return originalToCloned ? originalToCloned->lookupOrNull(original)
                              : original;

    IRMapping &spliceMap =
        originalToCloned ? *originalToCloned : inPlaceSplices;
    if (Value mapped = spliceMap.lookupOrNull(original))
      return mapped;

    for (Value operand : defOp->getOperands())
      resolve(operand, builder);

    builder.clone(*defOp, spliceMap);
    return spliceMap.lookup(original);
  }

public:
  // ApplyPartialDecisions is a hand-rolled PassWrapper (not generated via
  // GEN_PASS_DEF) defined in an anonymous namespace, so it needs an explicit
  // TypeID: PassWrapper's implicit fallback (TypeID::get<PassT>()) refuses
  // anonymous-namespace types when it can detect them, which some LLVM
  // builds only actually check under assertions/debug configurations.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyPartialDecisions)

  static std::unique_ptr<ApplyPartialDecisions>
  create(std::shared_ptr<StrategySearchNode> node,
         IRMapping &originalToCloned) {
    return std::unique_ptr<ApplyPartialDecisions>(
        new ApplyPartialDecisions(node, &originalToCloned));
  }

  static std::unique_ptr<ApplyPartialDecisions>
  createInPlace(std::shared_ptr<StrategySearchNode> node) {
    return std::unique_ptr<ApplyPartialDecisions>(
        new ApplyPartialDecisions(node, nullptr));
  }

  void runOnOperation() override {
    ModuleOp targetModule = getOperation();
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(targetModule.getBody());

    for (auto &[origAxis, origFactors] : node->getPartialDecisions()) {
      if (origFactors.empty()) {
        // Nothing decided for this axis yet; leave its pristine
        // representation untouched.
        continue;
      }

      Value targetAxisVal = resolve(origAxis, builder);
      if (!targetAxisVal) {
        origAxis.getDefiningOp()->emitError(
            "logical axis has no counterpart in the target module");
        return signalPassFailure();
      }
      auto targetAxis = cast<TypedValue<AxisTypeInterface>>(targetAxisVal);

      // Discover what to replace *before* building any new axis.factor
      // referencing targetAxis -- otherwise a newly-built factor (e.g. the
      // residual below) would itself show up as a use of targetAxis and be
      // mistaken for part of its current state.
      auto oldFactorOps = findAxisFactors(targetAxis);
      if (failed(oldFactorOps))
        return signalPassFailure();
      llvm::SmallVector<TypedValue<AxisFactorType>> oldFactors;
      for (AxisFactorOp oldFactorOp : *oldFactorOps)
        oldFactors.push_back(castTypedValue<AxisFactorType>(
            oldFactorOp.getResult(), "AxisFactorType"));

      llvm::SmallVector<TypedValue<AxisFactorType>> targetFactors;
      int64_t decidedExtent = 1;
      for (TypedValue<AxisFactorType> origFactor : origFactors) {
        auto origProvenance = getFactorProvenanceAxis(origFactor);
        if (failed(origProvenance)) {
          origFactor.getDefiningOp()->emitError(
              "failed to resolve provenance axis for a decided factor");
          return signalPassFailure();
        }
        Value targetProvenance = resolve(*origProvenance, builder);
        if (!targetProvenance) {
          origFactor.getDefiningOp()->emitError(
              "decided factor's provenance axis has no counterpart in the "
              "target module");
          return signalPassFailure();
        }
        int64_t extent = getFactorExtent(origFactor);
        decidedExtent *= extent;
        auto targetFactor =
            builder.create<AxisFactorOp>(origFactor.getLoc(), targetProvenance,
                                         extent, getFactorStride(origFactor));
        targetFactors.push_back(castTypedValue<AxisFactorType>(
            targetFactor.getResult(), "AxisFactorType"));
      }

      int64_t remainder = getAxisExtent(targetAxis) / decidedExtent;
      if (remainder > 1) {
        auto residualFactor = builder.create<AxisFactorOp>(
            targetAxisVal.getLoc(), targetAxisVal, remainder, 1);
        targetFactors.push_back(castTypedValue<AxisFactorType>(
            residualFactor.getResult(), "AxisFactorType"));
      }

      if (failed(replaceAxisFactors(oldFactors, targetFactors, builder)))
        return signalPassFailure();
    }
  }
};

// Not algorithmically fast! But we expect resonably small (10000k max,
// maybe) and easily divisible numbers.
static llvm::SmallVector<int64_t> uniquePrimeFactors(int64_t n) {
  llvm::SmallVector<int64_t> primes;
  int64_t thresh = 1;
  int64_t i = 2;
  while (i <= n) {
    if (n % i == 0) {
      if (i > thresh) {
        primes.push_back(i);
        thresh = i;
      }
      n /= i;
    } else {
      i++;
    }
  }
  return primes;
}

// A chunk peeled off a physical factor: `takenFactor` is bound to the axis
// being decided, `residualFactor` covers what's left of the physical factor
// and goes back into the available space.
struct ChunkDecision {
  SharedOpRef<AxisFactorOp> takenFactor;
  SharedOpRef<AxisFactorOp> residualFactor;
};

// Attempts to peel a chunk off of `physicalFactor`, using the first entry of
// `possibleChunks` (in the order given) whose extent evenly divides
// `physicalFactor`'s. Returns std::nullopt if no chunk in `possibleChunks`
// divides it.
static std::optional<ChunkDecision>
tryChunkFactor(SharedOpRef<AxisFactorOp> physicalFactor,
               llvm::ArrayRef<int64_t> possibleChunks, OpBuilder &builder) {
  // TODO do we have to swap available extent to owning things too?
  int64_t factorExtent = getFactorExtent(physicalFactor->get());
  int64_t chunkTaken = -1;
  for (auto chunk : possibleChunks) {
    if (chunk <= factorExtent && factorExtent % chunk == 0) {
      chunkTaken = chunk;
      break;
    }
  }
  if (chunkTaken == -1)
    return std::nullopt;

  int64_t residualExtent = factorExtent / chunkTaken;

  // We have a candidate that takes a chunk out of the physical factor,
  // from the high stride positions. Need to take the chunk factor,
  // and create a residual factor from the remaining extent.
  auto takenFactor =
      sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(
          createSubfactor(physicalFactor->get(), chunkTaken, residualExtent,
                          builder, physicalFactor->get().getLoc()));
  auto residualFactor =
      sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(
          createSubfactor(physicalFactor->get(), residualExtent, 1, builder,
                          physicalFactor->get().getLoc()));
  return ChunkDecision{takenFactor, residualFactor};
}

// Commits a chunk decision (see tryChunkFactor) to `node`: binds the taken
// chunk to the axis currently being decided, and swaps the physical factor
// it came from for its residual in the available space.
static void applyChunkDecision(StrategySearchNode &node,
                               SharedOpRef<AxisFactorOp> physicalFactor,
                               const ChunkDecision &decision) {
  node.addFactor(decision.takenFactor);
  node.removeAvailableSpace(physicalFactor);
  node.addAvailableSpace(decision.residualFactor);
}

// Consumes the entirety of `node`'s current axis's remaining extent by
// serializing it within the device (i.e. no further physical space needed).
static void applySerializeRemaining(StrategySearchNode &node,
                                    OpBuilder &builder, Location loc) {
  int64_t extentRemaining = node.getExtentRemaining();
  // spin up a fresh new owned reference to a within-device
  // axis, factor with extent equal to remaining
  DeviceLocalAxisOp localizationAxis =
      builder.create<DeviceLocalAxisOp>(loc, extentRemaining);
  auto factor = viewAxisAsFactor(localizationAxis, builder, loc);
  SharedOpRef<AxisFactorOp> owningFactor =
      sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(factor);
  SharedOpRef<Operation *> owningAxis =
      std::make_shared<OwningOpRef<Operation *>>(localizationAxis);
  node.addFactor(owningFactor);
  node.addSupportingIR(owningAxis);
}

class StrategyExplorer : public BeamSearchExplorerBase<StrategySearchNode> {
public:
  LogicalAxisOverlap &overlap;
  OpBuilder &builder;
  ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace;
  Location defaultLoc;
  bool logProgress;

  StrategyExplorer(LogicalAxisOverlap &overlap, OpBuilder &builder,
                   ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace,
                   Location defaultLoc, bool logProgress = false)
      : BeamSearchExplorerBase<StrategySearchNode>(), overlap(overlap),
        builder(builder), totalMeshSpace(totalMeshSpace),
        defaultLoc(defaultLoc), logProgress(logProgress) {}

  virtual std::vector<std::shared_ptr<StrategySearchNode>>
  generateCandidatesFromNode(
      std::shared_ptr<StrategySearchNode> node) override {
    // For the axis being worked on, decide if we want to
    // apply a sharding axis, pipeline (TODO), or place within
    // a device.
    const int64_t extentRemaining = node->getExtentRemaining();

    if (logProgress) {
      llvm::errs() << "distributed-search-strategies: parent at axis depth "
                   << node->getAxisIndex() << " of "
                   << node->getTotalAxisCount() << ", extent remaining "
                   << extentRemaining << ", "
                   << node->getAvailableSpace().size()
                   << " available physical factor(s)\n";
    }

    std::vector<std::shared_ptr<StrategySearchNode>> candidates;

    // Decision type 1: for factor of the physical space, see if we
    // can pull out a prime factor
    {
      auto possibleChunks = uniquePrimeFactors(extentRemaining);
      auto availableFactors = node->getAvailableSpace();
      for (auto physicalFactor : availableFactors) {
        auto decision = tryChunkFactor(physicalFactor, possibleChunks, builder);
        if (!decision)
          continue;
        auto child = node->makeChild();
        applyChunkDecision(*child, physicalFactor, *decision);
        child->considerNextAxis(overlap, totalMeshSpace, builder);
        if (logProgress) {
          llvm::errs() << "distributed-search-strategies:   child "
                       << candidates.size() << ": took chunk (extent "
                       << getFactorExtent(decision->takenFactor->get())
                       << ") from a physical factor of extent "
                       << getFactorExtent(physicalFactor->get())
                       << ", now at axis depth " << child->getAxisIndex()
                       << "\n";
        }
        candidates.push_back(child);
      }
    }

    // Decision type 2: serialize remaining
    {
      auto child = node->makeChild();
      applySerializeRemaining(*child, builder, defaultLoc);
      child->considerNextAxis(overlap, totalMeshSpace, builder);
      if (logProgress) {
        llvm::errs() << "distributed-search-strategies:   child "
                     << candidates.size() << ": serialized remaining extent "
                     << extentRemaining << ", now at axis depth "
                     << child->getAxisIndex() << "\n";
      }
      candidates.push_back(child);
    }
    return candidates;
  }
};

// Heuristically completes every remaining decision on a node so it can be
// lowered/scored as a full candidate. Implementations mutate `node` in
// place; callers are expected to pass an already-detached scratch clone
// (StrategySearchNode::makeChild()), never the live search node, since
// completion is meant to inform scoring without narrowing what the real
// search explores.
class StrategyCompleterBase {
public:
  virtual void complete(StrategySearchNode &node) = 0;
  virtual ~StrategyCompleterBase() = default;
};

// Completes axes in the same frozen order the search itself uses. For each
// axis, greedily takes the largest available chunk of free physical mesh
// space that still divides the remaining extent, repeating until the axis
// is fully divided; once no available factor can supply any usable chunk
// (out of free space, or no factor shares a divisor with what's left),
// serializes whatever extent remains within the device.
class StrategyInOrderCompleter : public StrategyCompleterBase {
  LogicalAxisOverlap &overlap;
  OpBuilder &builder;
  llvm::ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace;
  Location defaultLoc;

public:
  StrategyInOrderCompleter(
      LogicalAxisOverlap &overlap, OpBuilder &builder,
      llvm::ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace,
      Location defaultLoc)
      : overlap(overlap), builder(builder), totalMeshSpace(totalMeshSpace),
        defaultLoc(defaultLoc) {}

  void complete(StrategySearchNode &node) override {
    while (!node.finalized()) {
      int64_t extentRemaining = node.getExtentRemaining();
      assert(extentRemaining > 1 &&
             "considerNextAxis should have advanced past a done axis");

      // Descending order: unlike the explorer (which wants small, diverse
      // chunks to branch on), a one-shot completer should maximize
      // parallelism taken from each factor.
      auto possibleChunks = uniquePrimeFactors(extentRemaining);
      std::reverse(possibleChunks.begin(), possibleChunks.end());

      auto availableSnapshot = node.getAvailableSpace(); // stable copy
      bool took = false;
      for (auto physicalFactor : availableSnapshot) {
        if (auto decision =
                tryChunkFactor(physicalFactor, possibleChunks, builder)) {
          applyChunkDecision(node, physicalFactor, *decision);
          took = true;
          break;
        }
      }
      if (!took) {
        applySerializeRemaining(node, builder, defaultLoc);
      }
      node.considerNextAxis(overlap, totalMeshSpace, builder);
    }
  }
};

// Clones `originalModule`, applies `node`'s decisions to that clone, and
// runs the search's lowering pipeline on the result, via a standalone
// PassManager (not Pass::runPipeline, which requires its target to be
// nested under the operation the calling pass is currently processing --
// our clone is a disconnected top-level module). Reports pipeline success
// through `pipelineOk`.
//
// When `originalModule` carries distributed.logging_root, the clone also gets
// per-candidate logging (checkpoint IR dumps and collective counts).
// `suppressLogging` turns that off, for a caller replaying a node that was
// already logged when it was scored.
static OwningOpRef<ModuleOp> cloneAndApplyDecisions(
    ModuleOp originalModule, const std::shared_ptr<StrategySearchNode> &node,
    bool disableVerifier, llvm::StringRef dumpKernelModulesTo,
    llvm::StringRef actTarget, llvm::StringRef actWorkerDirRoot,
    llvm::StringRef actWorkerScriptDir, bool &pipelineOk,
    bool suppressLogging = false) {
  IRMapping mapper;
  OwningOpRef<ModuleOp> clonedModule(
      cast<ModuleOp>(originalModule->clone(mapper)));

  // The pipeline's passes learn which candidate they are working on from
  // distributed.logging_candidate_uid on their own module, the same way they
  // learn the log destination from distributed.logging_root. It is stamped
  // only when logging is active, so non-logging clones carry no logging
  // attributes.
  bool loggingActive =
      !suppressLogging && originalModule->hasAttr("distributed.logging_root");
  if (loggingActive) {
    Builder builder(originalModule.getContext());
    (*clonedModule)
        ->setAttr("distributed.logging_candidate_uid",
                  builder.getI64IntegerAttr(node->getUid()));
  }

  PassManager pm(originalModule.getContext(), ModuleOp::getOperationName());
  pm.enableVerifier(!disableVerifier);
  pm.addPass(ApplyPartialDecisions::create(node, mapper));
  {
    // Dumps the decided-but-not-yet-lowered clone, before any of
    // buildDistributedSearchLoweringPipeline's passes run. A no-op unless
    // logging is active, like every other checkpoint instance.
    DumpDistributedLoggingCheckpointPassOptions checkpointOptions;
    checkpointOptions.checkpointName = "00-after-decisions";
    pm.addPass(createDumpDistributedLoggingCheckpointPass(checkpointOptions));
  }
  // Kernel inlining and collective lowering run for every candidate (see
  // buildDistributedSearchLoweringPipeline): scoring needs the same fully
  // materialized IR shape (no kernel wrappers, no abstract collectives) that
  // ScheduleGraphAnalysis builds its cost graph from on the eventual winner,
  // so a candidate's simulated score can never disagree with how it would
  // actually be costed once chosen. Only schedule reordering (a physical IR
  // rewrite with nothing left for scoring to read afterward) is gated to the
  // winner alone, except that it also runs for every candidate when logging
  // is active (reorderForLogging), so each 03-final checkpoint is
  // post-reorder.
  buildDistributedSearchLoweringPipeline(
      pm, /*lowerLogicalAxes=*/false, dumpKernelModulesTo,
      /*finalizeSchedule=*/false, actTarget, actWorkerDirRoot,
      actWorkerScriptDir, /*reorderForLogging=*/loggingActive);
  pipelineOk = succeeded(pm.run(*clonedModule));
  return clonedModule;
}

// Prints a debug dump of `module`, the cloned/partially-rewritten IR for one
// search candidate, distinguishing what kind of dump this is (`header`) and,
// when available, its search score and a one-line `note` on how the score
// was computed. `uid`/`parentUid` identify the StrategySearchNode this
// candidate came from, so a chain of dumps can be reassembled into the
// search tree that produced them.
static void dumpSearchModule(llvm::StringRef header, ModuleOp module,
                             bool pipelineOk, uint64_t uid,
                             std::optional<uint64_t> parentUid,
                             std::optional<double> score = std::nullopt,
                             llvm::StringRef note = "") {
  llvm::errs() << "// " << header << " (" << (pipelineOk ? "ok" : "FAILED");
  if (score)
    llvm::errs() << ", score=" << *score;
  llvm::errs() << "):\n";
  llvm::errs() << "// uid=" << uid;
  if (parentUid)
    llvm::errs() << " parent_uid=" << *parentUid;
  llvm::errs() << "\n";
  if (!note.empty()) {
    std::string commented = note.str();
    for (size_t at = commented.find('\n'); at != std::string::npos;
         at = commented.find('\n', at + 4))
      commented.replace(at, 1, "\n// ");
    llvm::errs() << "// " << commented << "\n";
  }
  module.print(llvm::errs());
  llvm::errs() << "\n";
}

// One line summarizing a feasible timeline: the makespan and how much of it
// is spent communicating vs. computing (ScheduleSimulator.h's
// TimelineResult), the same breakdown distributed-print-schedule-cost
// reports for a single module.
static std::string describeTimeline(const TimelineResult &result,
                                    size_t numNodes) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "schedule cost: makespan=" << result.makespan << " over " << numNodes
     << " nodes (communicating=" << result.totalCommunicationTime
     << " computing=" << result.totalComputeTime
     << " utilization=" << result.averageComputeUtilization
     << " idle=" << result.totalComputeIdleTime << ")";
  return text;
}

// The ScheduleGraph and TimelineResult a feasible scoreLowered() call builds
// to compute its makespan, kept so per-candidate logging can serialize them.
struct ScoredSchedule {
  ScheduleGraph graph;
  TimelineResult timeline;
};

// Serializes `value` to `path` so a concurrent reader never observes a
// partially-written file. Errors are reported on `diagnosticModule`.
static LogicalResult writeJsonAtomically(ModuleOp diagnosticModule,
                                         llvm::StringRef path,
                                         const llvm::json::Value &value) {
  std::string text;
  llvm::raw_string_ostream(text) << value;

  // Write to a sibling .tmp file, then rename over `path`.
  llvm::SmallString<128> tmpPath(path);
  tmpPath += ".tmp";
  std::error_code ec;
  {
    llvm::raw_fd_ostream os(tmpPath, ec, llvm::sys::fs::OF_None);
    if (ec)
      return diagnosticModule.emitError()
             << "distributed-search-strategies: failed to open '" << tmpPath
             << "' for writing: " << ec.message();
    os << text;
  }
  if (auto renameEc = llvm::sys::fs::rename(tmpPath, path))
    return diagnosticModule.emitError()
           << "distributed-search-strategies: failed to publish '" << path
           << "': " << renameEc.message();
  return success();
}

// A feasible candidate's compute time split between real ACT accelerator
// dispatches and everything else (ordinary ops, plus ACT's zero-cost
// stage/fetch calls, which add no time to either bucket).
struct ComputeTimeSplit {
  double acceleratorTime = 0;
  double cpuTime = 0;
  int64_t numAcceleratorLaunches = 0;
};

static ComputeTimeSplit splitComputeTime(const ScheduleGraph &graph,
                                         const TimelineResult &timeline) {
  ComputeTimeSplit split;
  for (size_t i = 0, e = graph.nodes.size(); i < e; ++i) {
    const ScheduleEvent &node = graph.nodes[i];
    if (node.kind != ScheduleEventKind::Compute)
      continue;
    double duration = timeline.finish[i] - timeline.start[i];
    if (node.isAcceleratorLaunch) {
      split.acceleratorTime += duration;
      ++split.numAcceleratorLaunches;
    } else {
      split.cpuTime += duration;
    }
  }
  return split;
}

// JSON `null` for a non-finite double (this model only ever produces
// -infinity, for an unscored/infeasible candidate's heuristic_score): JSON
// has no infinity literal, and llvm::json's own serializer (OStream::value)
// prints a non-finite double via a plain "%g", yielding literal "-inf" text
// that is not valid JSON.
static llvm::json::Value jsonDoubleOrNull(double value) {
  return std::isfinite(value) ? llvm::json::Value(value)
                              : llvm::json::Value(nullptr);
}

// JSON `null` for an absent collectives count. The counts are stamped on a
// candidate's clone only once its pipeline reaches the counting checkpoints
// (see DumpDistributedLoggingCheckpointPass's count-collectives-as option), so
// a pipeline failure leaves them unset.
static llvm::json::Value jsonInt64OrNull(std::optional<int64_t> value) {
  return value ? llvm::json::Value(*value) : llvm::json::Value(nullptr);
}

// Writes Candidates/<uid>/info.json for one scored candidate, plus
// schedule.json when `schedule` is present (a feasible, fully simulated
// candidate).
//
// Called for every scored candidate, including a pipeline failure or an
// infeasible collective graph. Both have no `schedule`, so their info.json
// has no `summary` block and no schedule.json is written.
// `numCollectivesBefore`/`numCollectivesAfter` are absent when the pipeline
// did not reach both counting checkpoints (see
// DumpDistributedLoggingCheckpointPass), as always happens on a pipeline
// failure.
static LogicalResult
writeCandidateLog(ModuleOp originalModule, llvm::StringRef loggingRoot,
                  const StrategySearchNode &node, bool pipelineOk,
                  double heuristicScore,
                  const std::optional<ScoredSchedule> &schedule,
                  std::optional<int64_t> numCollectivesBefore,
                  std::optional<int64_t> numCollectivesAfter) {
  llvm::SmallString<128> candidateDir(loggingRoot);
  llvm::sys::path::append(candidateDir, "Candidates",
                          llvm::Twine(node.getUid()));
  if (auto ec = llvm::sys::fs::create_directories(candidateDir))
    return originalModule.emitError()
           << "distributed-search-strategies: failed to create '"
           << candidateDir << "': " << ec.message();

  llvm::json::Object info;
  info["uid"] = static_cast<int64_t>(node.getUid());
  info["parent_uid"] =
      node.getParentUid()
          ? llvm::json::Value(static_cast<int64_t>(*node.getParentUid()))
          : llvm::json::Value(nullptr);
  info["pipeline_ok"] = pipelineOk;
  info["heuristic_score"] = jsonDoubleOrNull(heuristicScore);
  info["num_collectives_before"] = jsonInt64OrNull(numCollectivesBefore);
  info["num_collectives_after"] = jsonInt64OrNull(numCollectivesAfter);

  if (schedule) {
    ComputeTimeSplit split =
        splitComputeTime(schedule->graph, schedule->timeline);
    llvm::json::Object summary;
    summary["num_act_kernel_matches"] = split.numAcceleratorLaunches;
    summary["accelerator_compute_time"] = split.acceleratorTime;
    summary["cpu_compute_time"] = split.cpuTime;
    summary["idle_time"] = schedule->timeline.totalComputeIdleTime;
    summary["communication_time"] = schedule->timeline.totalCommunicationTime;
    summary["total_time"] = schedule->timeline.makespan;
    info["summary"] = std::move(summary);
  }

  llvm::SmallString<128> infoPath(candidateDir);
  llvm::sys::path::append(infoPath, "info.json");
  if (failed(writeJsonAtomically(originalModule, infoPath,
                                 llvm::json::Value(std::move(info)))))
    return failure();

  if (!schedule)
    return success();

  llvm::json::Array nodes;
  for (size_t i = 0, e = schedule->graph.nodes.size(); i < e; ++i) {
    const ScheduleEvent &event = schedule->graph.nodes[i];
    llvm::json::Array predecessors;
    for (size_t pred : event.predecessors)
      predecessors.push_back(static_cast<int64_t>(pred));
    llvm::json::Object nodeObj;
    nodeObj["index"] = static_cast<int64_t>(i);
    nodeObj["kind"] = toString(event.kind);
    nodeObj["description"] = describeEvent(event);
    nodeObj["start"] = schedule->timeline.start[i];
    nodeObj["finish"] = schedule->timeline.finish[i];
    nodeObj["predecessors"] = std::move(predecessors);
    nodes.push_back(std::move(nodeObj));
  }
  llvm::json::Object scheduleJson;
  scheduleJson["makespan"] = schedule->timeline.makespan;
  scheduleJson["nodes"] = std::move(nodes);

  llvm::SmallString<128> schedulePath(candidateDir);
  llvm::sys::path::append(schedulePath, "schedule.json");
  return writeJsonAtomically(originalModule, schedulePath,
                             llvm::json::Value(std::move(scheduleJson)));
}

class StrategyScorer : public BeamSearchScorerBase<StrategySearchNode> {
  ModuleOp originalModule;
  bool dumpCandidates;
  bool disableVerifier;
  std::string dumpKernelModulesTo;
  std::string actTarget;
  std::string actWorkerDirRoot;
  std::string actWorkerScriptDir;
  bool failOnBadCandidate;
  bool &sawBadCandidate;
  BeamSearchQueueBase<StrategySearchNode> &queue;
  StrategyCompleterBase &completer;
  size_t numScored = 0;
  size_t numPlanFailures = 0;
  std::string firstPlanFailure;
  // Candidates whose nested PassManager in cloneAndApplyDecisions failed
  // (pipelineOk=false). numPlanFailures separately counts candidates that
  // lowered but scored -infinity because a collective could not be planned.
  size_t numPipelineErrors = 0;

public:
  StrategyScorer(ModuleOp originalModule, bool dumpCandidates,
                 bool disableVerifier, llvm::StringRef dumpKernelModulesTo,
                 llvm::StringRef actTarget, llvm::StringRef actWorkerDirRoot,
                 llvm::StringRef actWorkerScriptDir, bool failOnBadCandidate,
                 bool &sawBadCandidate,
                 BeamSearchQueueBase<StrategySearchNode> &queue,
                 StrategyCompleterBase &completer)
      : originalModule(originalModule), dumpCandidates(dumpCandidates),
        disableVerifier(disableVerifier),
        dumpKernelModulesTo(dumpKernelModulesTo), actTarget(actTarget),
        actWorkerDirRoot(actWorkerDirRoot),
        actWorkerScriptDir(actWorkerScriptDir),
        failOnBadCandidate(failOnBadCandidate),
        sawBadCandidate(sawBadCandidate), queue(queue), completer(completer) {}

  // Search statistics: candidates scored, how many of those lowered but had a
  // collective that could not be planned, and the first such reason.
  size_t getNumScored() const { return numScored; }
  size_t getNumPlanFailures() const { return numPlanFailures; }
  const std::string &getFirstPlanFailure() const { return firstPlanFailure; }
  size_t getNumPipelineErrors() const { return numPipelineErrors; }

  // Scores a module that lowered successfully, describing the score in
  // `note`. A feasible candidate's graph and timeline are returned in
  // `schedule` for logging; an infeasible module scores -infinity and leaves
  // `schedule` unset.
  double scoreLowered(ModuleOp lowered, std::string &note,
                      std::optional<ScoredSchedule> &schedule) {
    // `lowered` is a disposable, one-shot clone with no surrounding
    // AnalysisManager to cache through (see cloneAndApplyDecisions). The graph
    // is built directly with buildScheduleGraph, which returns it by value so
    // it can be moved into `schedule` below.
    ScheduleGraph graph = buildScheduleGraph(lowered);
    if (!graph.feasible) {
      if (numPlanFailures++ == 0)
        firstPlanFailure = graph.failureReason;
      note = "collective planning failed: " + graph.failureReason;
      return -std::numeric_limits<double>::infinity();
    }
    // The graph build has already found the unique physical mesh.
    PhysicalMeshOp physicalMesh = *findUniquePhysicalMesh(lowered);
    MeshCostParams mesh = meshCostParamsFromPhysicalMesh(physicalMesh);
    DeviceCostParams device = deviceCostParamsFromPhysicalMesh(physicalMesh);
    TimelineResult result = simulateScheduleGraph(graph, mesh, device);
    note = describeTimeline(result, graph.nodes.size());
    double score = 0.0 - result.makespan;
    schedule = ScoredSchedule{std::move(graph), std::move(result)};
    return score;
  }

  // Applies and lowers the candidate's decisions on a clone, then scores it.
  // Pipeline:
  // - StrategyCompleterBase (StrategyInOrderCompleter) : heuristically
  //   complete the remaining decisions on the node itself, before any IR is
  //   cloned
  // - ApplyPartialDecisions : materialize the (now fully-decided) decisions
  //   onto a clone
  // - Lowering pipeline (buildDistributedSearchLoweringPipeline) : lowers
  //   the fully-decided clone to atomic distributed.Collective ops
  // - ScheduleGraph + ScheduleSimulator : builds the lowered clone's event DAG
  //   and simulates it to a predicted makespan
  //
  // The score is the negated predicted makespan (higher is better, as the
  // beam maximizes): a discrete-event, max-min fair-share timeline simulation
  // over the candidate's compute ops and decomposed collective steps (see
  // ScheduleGraph.h and ScheduleSimulator.h), so independent collectives on
  // different mesh axes and compute overlapping communication are credited
  // with running concurrently rather than summed as if serialized. The mesh
  // parameters come from the physical mesh's metadata, else abstract
  // defaults.
  //
  // A candidate is infeasible, and scores -infinity, if its lowering pipeline
  // fails or any of its collectives cannot be planned.
  double score(const std::shared_ptr<StrategySearchNode> &node) override {
    // Complete decisions on a throwaway clone so scoring can see a fully
    // decided candidate without narrowing what the real search explores.
    // clone() keeps `node`'s uid: the copy is discarded after scoring and is
    // never itself a node in the search tree.
    std::shared_ptr<StrategySearchNode> completedNode = node->clone();
    completer.complete(*completedNode);

    bool pipelineOk;
    OwningOpRef<ModuleOp> clonedModule = cloneAndApplyDecisions(
        originalModule, completedNode, disableVerifier, dumpKernelModulesTo,
        actTarget, actWorkerDirRoot, actWorkerScriptDir, pipelineOk);

    ++numScored;
    if (!pipelineOk)
      ++numPipelineErrors;
    std::string note;
    std::optional<ScoredSchedule> schedule;
    double result = pipelineOk ? scoreLowered(*clonedModule, note, schedule)
                               : -std::numeric_limits<double>::infinity();

    // Stamped by the 02-before-collectives and 03-final checkpoints (see
    // DumpDistributedLoggingCheckpointPass), so absent on a pipeline failure
    // or when logging is off.
    std::optional<int64_t> numCollectivesBefore, numCollectivesAfter;
    if (pipelineOk) {
      if (auto attr = (*clonedModule)
                          ->getAttrOfType<IntegerAttr>(
                              "distributed.logging_num_collectives_before"))
        numCollectivesBefore = attr.getInt();
      if (auto attr = (*clonedModule)
                          ->getAttrOfType<IntegerAttr>(
                              "distributed.logging_num_collectives_after"))
        numCollectivesAfter = attr.getInt();
    }

    // Every scored candidate is logged, not only finalized ones, so the search
    // tree can be reconstructed afterward (see writeCandidateLog).
    if (auto loggingRoot = originalModule->getAttrOfType<StringAttr>(
            "distributed.logging_root"))
      if (failed(writeCandidateLog(
              originalModule, loggingRoot.getValue(), *node, pipelineOk, result,
              schedule, numCollectivesBefore, numCollectivesAfter))) {
        // A write failure (already reported by writeCandidateLog) indicates a
        // broken logging destination, e.g. a full or unwritable disk. Abort
        // the whole search, as fail-on-bad-candidate does for a pipeline
        // failure, since continuing would silently drop logged data.
        sawBadCandidate = true;
        queue.abort();
      }

    if (dumpCandidates)
      dumpSearchModule("Search candidate", *clonedModule, pipelineOk,
                       node->getUid(), node->getParentUid(), result, note);

    if (!pipelineOk && failOnBadCandidate && !sawBadCandidate) {
      sawBadCandidate = true;
      // Always show this one candidate's IR, even without dump-candidates,
      // since it's the whole reason failOnBadCandidate exists: surfacing the
      // bug immediately instead of it being masked by the search discarding
      // the candidate and moving on.
      if (!dumpCandidates)
        dumpSearchModule("Search candidate", *clonedModule, pipelineOk,
                         node->getUid(), node->getParentUid(), result, note);
      originalModule.emitError()
          << "distributed-search-strategies: failing outright because "
             "fail-on-bad-candidate is set and a search candidate's "
             "lowering pipeline failed (see the printed candidate IR above)";
      // Discard the rest of the beam so the search stops now rather than
      // exploring and scoring every remaining candidate first.
      queue.abort();
    }

    return result;
  }
};

// Writes <loggingRoot>/Main/info.json once the search has finished:
// per-compilation counts, and which candidate (if any) won.
static LogicalResult writeMainInfoJson(
    ModuleOp moduleOp, llvm::StringRef loggingRoot, size_t numLogicalAxes,
    const StrategyScorer &scorer,
    const std::shared_ptr<StrategySearchNode> &best,
    llvm::ArrayRef<std::shared_ptr<StrategySearchNode>> finalized) {
  llvm::json::Object info;
  info["num_logical_axes"] = static_cast<int64_t>(numLogicalAxes);
  if (auto moduleName = moduleOp->getAttrOfType<StringAttr>(
          "distributed.logging_module_name"))
    info["input_module_name"] = moduleName.getValue();
  else
    info["input_module_name"] = nullptr;
  if (auto commit =
          moduleOp->getAttrOfType<StringAttr>("distributed.logging_commit"))
    info["commit_or_version"] = commit.getValue();
  else
    info["commit_or_version"] = nullptr;
  info["best_candidate_uid"] =
      best ? llvm::json::Value(static_cast<int64_t>(best->getUid()))
           : llvm::json::Value(nullptr);
  // The candidates the search finished with (every axis decided and kept in
  // the final beam), in the driver's order.
  llvm::json::Array finalizedUids;
  for (const auto &node : finalized)
    finalizedUids.push_back(static_cast<int64_t>(node->getUid()));
  info["finalized_candidate_uids"] = std::move(finalizedUids);
  info["num_candidates_scored"] = static_cast<int64_t>(scorer.getNumScored());
  info["num_candidates_pipeline_errored"] =
      static_cast<int64_t>(scorer.getNumPipelineErrors());
  info["num_candidates_infeasible"] =
      static_cast<int64_t>(scorer.getNumPlanFailures());

  llvm::SmallString<128> finalPath(loggingRoot);
  llvm::sys::path::append(finalPath, "Main", "info.json");
  return writeJsonAtomically(moduleOp, finalPath,
                             llvm::json::Value(std::move(info)));
}

// Writes <loggingRoot>/Main/logical_axes.json: every logical/device-local
// axis in the module, keyed by a distributed.logging_id that this walk mints
// and stamps on each axis op.
//
// Axis ops are created by several passes that do not know whether logging is
// active, so ids are minted here, gated like every other logging output.
// logging_id therefore exists on a module only when logging is on.
// resolvePartitioningAxesToLoggingIds relies on this having already stamped
// every axis op.
//
// Broader than findAllLogicalAxes, which collects only LogicalMeshAxesOps with
// live users (the axes the search decides over). A DeviceLocalAxisOp is never
// searched over but still occupies a slot in a tensor's sharding index space,
// so it is listed here too.
static LogicalResult writeLogicalAxesJson(ModuleOp moduleOp,
                                          llvm::StringRef loggingRoot) {
  llvm::json::Array axesJson;
  int64_t nextLoggingId = 1;
  Builder builder(moduleOp.getContext());
  moduleOp.walk([&](Operation *op) {
    llvm::StringRef kind;
    uint64_t extent;
    if (auto axesOp = dyn_cast<LogicalMeshAxesOp>(op)) {
      kind = "logical_mesh";
      extent = axesOp.getExtent();
    } else if (auto localOp = dyn_cast<DeviceLocalAxisOp>(op)) {
      kind = "device_local";
      extent = localOp.getExtent();
    } else {
      return;
    }
    int64_t loggingId = nextLoggingId++;
    op->setAttr("distributed.logging_id", builder.getI64IntegerAttr(loggingId));
    llvm::json::Object entry;
    entry["logging_id"] = loggingId;
    entry["extent"] = static_cast<int64_t>(extent);
    entry["kind"] = kind;
    axesJson.push_back(std::move(entry));
  });

  llvm::json::Object root;
  root["axes"] = std::move(axesJson);
  llvm::SmallString<128> path(loggingRoot);
  llvm::sys::path::append(path, "Main", "logical_axes.json");
  return writeJsonAtomically(moduleOp, path,
                             llvm::json::Value(std::move(root)));
}

// Resolves each of `kernelOp`'s partitioning_axes operands (its ordered
// logical-axis index space, see DistributedKernelOp's description) to the
// distributed.logging_id of the LogicalMeshAxesOp/DeviceLocalAxisOp behind it.
//
// Assumes the def-chain ClusterDistributedKernels.cpp builds for every
// operand (kernelPartitioningAxes there): an axis.product wrapping exactly
// one axis.factor, whose axis operand is the axis op itself. That is the only
// site that builds partitioning_axes at the point this pass walks the module
// (MergeAdjacentTrivialKernels, the other creator of DistributedKernelOp, runs
// later in the per-candidate lowering pipeline), so the chain is asserted.
// Also requires writeLogicalAxesJson to have already stamped logging_id.
static llvm::SmallVector<int64_t>
resolvePartitioningAxesToLoggingIds(DistributedKernelOp kernelOp) {
  llvm::SmallVector<int64_t> loggingIds;
  loggingIds.reserve(kernelOp.getPartitioningAxes().size());
  for (Value axisOperand : kernelOp.getPartitioningAxes()) {
    auto factorGroup = cast<TypedValue<FactorGroupType>>(axisOperand);
    auto factors = getProductProvenanceFactors(factorGroup);
    assert(succeeded(factors) &&
           "a DistributedKernelOp's partitioning_axes operand is always "
           "built via axis.product (viewFactorsAsProduct)");
    assert(factors->size() == 1 &&
           "a kernel's own partitioning_axes group always wraps exactly one "
           "factor (ClusterDistributedKernels.cpp's kernelPartitioningAxes)");
    auto axisValue = getFactorProvenanceAxis((*factors)[0]);
    assert(succeeded(axisValue) &&
           "a partitioning-axis factor is always built via axis.factor "
           "(viewAxisAsFactor)");
    Operation *axisDefOp = axisValue->getDefiningOp();
    auto loggingId =
        axisDefOp->getAttrOfType<IntegerAttr>("distributed.logging_id");
    assert(loggingId && "writeLogicalAxesJson stamps every LogicalMeshAxesOp/"
                        "DeviceLocalAxisOp in the module before this walk "
                        "runs, and the two are only ever called together, "
                        "in that order, from runOnOperation");
    loggingIds.push_back(loggingId.getInt());
  }
  return loggingIds;
}

// Resolves one IndexedTensorShardingAttr's index-space fields
// (dim_partitioning_axes and unreduced_axes, both indices into the owning
// DistributedKernelOp's partitioning_axes list) to logging ids.
static llvm::json::Object
shardingToJson(IndexedTensorShardingAttr sharding,
               llvm::ArrayRef<int64_t> axisIndexToLoggingId) {
  llvm::json::Array dims;
  for (DenseI64ArrayAttr dimAxes : sharding.getDimPartitioningAxes()) {
    llvm::json::Array dimIds;
    for (int64_t idx : dimAxes.asArrayRef())
      dimIds.push_back(axisIndexToLoggingId[idx]);
    dims.push_back(std::move(dimIds));
  }
  llvm::json::Array unreducedIds;
  for (int64_t idx : sharding.getUnreducedAxes().asArrayRef())
    unreducedIds.push_back(axisIndexToLoggingId[idx]);
  llvm::json::Object obj;
  obj["dims"] = std::move(dims);
  obj["unreduced_axes"] = std::move(unreducedIds);
  return obj;
}

static llvm::json::Array
shardingsToJson(IndexedTensorShardingPerValueAttr perValue,
                llvm::ArrayRef<int64_t> axisIndexToLoggingId) {
  llvm::json::Array result;
  for (IndexedTensorShardingAttr sharding : perValue.getShardings())
    result.push_back(shardingToJson(sharding, axisIndexToLoggingId));
  return result;
}

// Writes <loggingRoot>/Main/op_axis_tags.json: the op-to-logical-axis
// correspondence that ClusterDistributedKernels fixes before search starts. A
// candidate only decides which physical factor binds to a logical axis, so one
// walk over the original module covers every candidate.
static LogicalResult writeOpAxisTagsJson(ModuleOp moduleOp,
                                         llvm::StringRef loggingRoot) {
  llvm::json::Array opsJson;
  llvm::DenseMap<Operation *, llvm::SmallVector<int64_t>> kernelAxisCache;
  moduleOp.walk([&](Operation *op) {
    auto argShardings = op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
        "distributed.argument_shardings");
    auto outShardings = op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
        "distributed.output_shardings");
    if (!argShardings && !outShardings)
      return;

    // ClusterDistributedKernels.cpp is the only site that sets these
    // attributes, and only on ops inside a DistributedKernelOp body.
    auto kernelOp = op->getParentOfType<DistributedKernelOp>();
    assert(kernelOp && "distributed.argument_shardings/output_shardings only "
                       "ever appears on a DistributedKernelOp body member");
    Operation *kernelOpPtr = kernelOp.getOperation();

    auto cacheIt = kernelAxisCache.find(kernelOpPtr);
    if (cacheIt == kernelAxisCache.end())
      cacheIt = kernelAxisCache
                    .try_emplace(kernelOpPtr,
                                 resolvePartitioningAxesToLoggingIds(kernelOp))
                    .first;
    llvm::ArrayRef<int64_t> axisIndexToLoggingId = cacheIt->second;

    llvm::json::Object entry;
    entry["location"] = mlir::debugString(op->getLoc());
    if (argShardings)
      entry["argument_shardings"] =
          shardingsToJson(argShardings, axisIndexToLoggingId);
    if (outShardings)
      entry["output_shardings"] =
          shardingsToJson(outShardings, axisIndexToLoggingId);
    opsJson.push_back(std::move(entry));
  });

  llvm::json::Object root;
  root["ops"] = std::move(opsJson);
  llvm::SmallString<128> path(loggingRoot);
  llvm::sys::path::append(path, "Main", "op_axis_tags.json");
  return writeJsonAtomically(moduleOp, path,
                             llvm::json::Value(std::move(root)));
}

struct DistributedSearchStrategiesPass
    : public impl::DistributedSearchStrategiesPassBase<
          DistributedSearchStrategiesPass> {
  using DistributedSearchStrategiesPassBase::
      DistributedSearchStrategiesPassBase;

  void runOnOperation() override {
    // Mirrors LowerKernelsToExecutablePass's own validation of these same
    // three options (see there), but enforced independently here, before the
    // search loop starts: a config mistake reaching cloneAndApplyDecisions
    // during real per-candidate scoring would otherwise surface as
    // pipelineOk=false for every candidate (StrategyScorer::score treats that
    // as "infeasible, score -infinity", not a hard failure), burying one
    // clear config error under a search that silently finds no viable
    // candidate instead of failing loudly up front.
    bool anyActDispatchOption = !actTarget.empty() ||
                                !actWorkerDirRoot.empty() ||
                                !actWorkerScriptDir.empty();
    bool allActDispatchOptions = !actTarget.empty() &&
                                 !actWorkerDirRoot.empty() &&
                                 !actWorkerScriptDir.empty();
    if (anyActDispatchOption && !allActDispatchOptions) {
      getOperation()->emitError()
          << "act-target, act-worker-dir-root, and act-worker-script-dir "
             "must all be set together to activate real ACT dispatch during "
             "search scoring (got act-target='"
          << actTarget << "', act-worker-dir-root='" << actWorkerDirRoot
          << "', act-worker-script-dir='" << actWorkerScriptDir << "')";
      signalPassFailure();
      return;
    }

    ModuleOp moduleOp = getOperation();
    mlir::OpBuilder builder(moduleOp.getContext());
    builder.clearInsertionPoint();

    // Walks the original module before any candidate is cloned from it. The
    // op-to-logical-axis correspondence these files record is the same for
    // every candidate, so it is written once here.
    //
    // Assumes kernel-body ops still carry the distributed.argument_shardings/
    // output_shardings that the shardy-to-distributed pipeline stamped; those
    // are stripped only per candidate, inside the lowering pipeline.
    // writeOpAxisTagsJson reads the logging_id that writeLogicalAxesJson
    // stamps, so the two must run in this order.
    if (auto loggingRoot =
            moduleOp->getAttrOfType<StringAttr>("distributed.logging_root")) {
      if (failed(writeLogicalAxesJson(moduleOp, loggingRoot.getValue())))
        return signalPassFailure();
      if (failed(writeOpAxisTagsJson(moduleOp, loggingRoot.getValue())))
        return signalPassFailure();
    }

    auto overlap = LogicalAxisOverlap(moduleOp);

    auto logicalAxes = findAllLogicalAxes(moduleOp);
    llvm::SmallVector<SharedOpRef<AxisFactorOp>> physicalAxes =
        findAllPhysicalAxes(moduleOp, builder, moduleOp.getLoc());

    // TODO: order axis search order by importance
    auto axes =
        std::make_shared<const LogicalAxisOrder>(std::move(logicalAxes));
    auto decisions = AxisMappingReplayTree::makeRoot();

    auto initialNode = std::make_shared<StrategySearchNode>(axes, decisions);
    initialNode->setupNextAxis(overlap, physicalAxes, builder);

    std::function<void(llvm::ArrayRef<std::shared_ptr<StrategySearchNode>>)>
        progressLogger;
    if (logProgress) {
      std::size_t totalAxes = axes->size();
      progressLogger =
          [totalAxes](
              llvm::ArrayRef<std::shared_ptr<StrategySearchNode>> generation) {
            std::size_t shallowest = generation.front()->getAxisIndex();
            std::size_t deepest = shallowest;
            for (const auto &node : generation) {
              shallowest = std::min(shallowest, node->getAxisIndex());
              deepest = std::max(deepest, node->getAxisIndex());
            }
            llvm::errs() << "distributed-search-strategies: beam turnover ("
                         << generation.size() << " candidate(s)), axis depth "
                         << shallowest << "-" << deepest << " of " << totalAxes
                         << "\n";
          };
    }
    BeamSearchBreadthFirstQueue<StrategySearchNode> queue(beamSize,
                                                          progressLogger);
    queue.push(initialNode);
    StrategyExplorer explorer(overlap, builder, physicalAxes, moduleOp.getLoc(),
                              logProgress);
    StrategyInOrderCompleter completer(overlap, builder, physicalAxes,
                                       moduleOp.getLoc());
    bool sawBadCandidate = false;
    StrategyScorer scorer(moduleOp, dumpCandidates, disableVerifier,
                          dumpKernelModulesTo, actTarget, actWorkerDirRoot,
                          actWorkerScriptDir, failOnBadCandidate,
                          sawBadCandidate, queue, completer);

    BeamSearchDriver<StrategySearchNode> driver(queue, scorer, explorer);
    driver.run();

    if (sawBadCandidate) {
      signalPassFailure();
      return;
    }

    if (logProgress)
      llvm::errs() << "distributed-search-strategies: scored "
                   << scorer.getNumScored() << " candidates\n";

    // An infeasible collective scores its candidate -infinity, so a search
    // that hits one everywhere would silently degrade; report the count once.
    if (scorer.getNumPlanFailures() > 0)
      moduleOp.emitRemark()
          << "distributed-search-strategies: " << scorer.getNumPlanFailures()
          << " of " << scorer.getNumScored()
          << " scored candidates had a collective that could not be planned "
             "(first reason: "
          << scorer.getFirstPlanFailure() << ")";

    // A hard pipeline failure (a bug or config error) is a different failure
    // mode than an infeasible collective above, so it is counted separately.
    if (scorer.getNumPipelineErrors() > 0)
      moduleOp.emitRemark()
          << "distributed-search-strategies: " << scorer.getNumPipelineErrors()
          << " of " << scorer.getNumScored()
          << " scored candidates had a hard lowering-pipeline failure";

    // getBest() returns a field BeamSearchDriver already tracks during run(),
    // so it can be read here, ahead of the dump logic below, for
    // Main/info.json.
    auto best = driver.getBest();

    if (auto loggingRoot =
            moduleOp->getAttrOfType<StringAttr>("distributed.logging_root"))
      if (failed(writeMainInfoJson(moduleOp, loggingRoot.getValue(),
                                   axes->size(), scorer, best,
                                   driver.getFinalized())))
        return signalPassFailure();

    // Only dumped once the search is complete, so finalized candidates aren't
    // interleaved with the in-progress ones dumpCandidates prints during the
    // search itself. Each node here was already logged when scored during the
    // search, so this replay passes suppressLogging=true: it exists only for
    // dumpSearchModule's stderr output and must not overwrite the real
    // Candidates/<uid>/ log.
    if (dumpFinalized) {
      for (const auto &node : driver.getFinalized()) {
        bool pipelineOk;
        OwningOpRef<ModuleOp> clonedModule = cloneAndApplyDecisions(
            moduleOp, node, disableVerifier, /*dumpKernelModulesTo=*/"",
            actTarget, actWorkerDirRoot, actWorkerScriptDir, pipelineOk,
            /*suppressLogging=*/true);
        std::string note;
        std::optional<ScoredSchedule> schedule;
        if (pipelineOk)
          scorer.scoreLowered(*clonedModule, note, schedule);
        dumpSearchModule("Finalized search candidate", *clonedModule,
                         pipelineOk, node->getUid(), node->getParentUid(),
                         node->score, note);
      }
    }

    // As with dumpFinalized above, `best` was already logged during scoring.
    if (dumpBest) {
      if (best) {
        bool pipelineOk;
        OwningOpRef<ModuleOp> clonedModule = cloneAndApplyDecisions(
            moduleOp, best, disableVerifier, /*dumpKernelModulesTo=*/"",
            actTarget, actWorkerDirRoot, actWorkerScriptDir, pipelineOk,
            /*suppressLogging=*/true);
        std::string note;
        std::optional<ScoredSchedule> schedule;
        if (pipelineOk)
          scorer.scoreLowered(*clonedModule, note, schedule);
        dumpSearchModule("Best search candidate", *clonedModule, pipelineOk,
                         best->getUid(), best->getParentUid(), best->score,
                         note);
      } else {
        llvm::errs() << "// Best search candidate: none found (no finalized "
                        "candidates)\n";
      }
    }

    // An infeasible finalized candidate still scores -infinity (see
    // StrategyScorer::score) rather than being dropped, so it can still win
    // when nothing else finalizes; treat that the same as no candidate at all
    // rather than committing known-bad decisions.
    if (!best || best->score == -std::numeric_limits<double>::infinity()) {
      moduleOp.emitError()
          << "distributed-search-strategies: no viable candidate found "
             "(every explored candidate failed lowering or collective "
             "planning)";
      return signalPassFailure();
    }

    // Everything above only explored decisions against disposable clones;
    // bind the winning decisions onto the real module now, in place, so no
    // LogicalMeshAxisType this pass owns survives into later pipeline stages.
    OpPassManager applyPm(ModuleOp::getOperationName());
    applyPm.addPass(ApplyPartialDecisions::createInPlace(best));
    if (failed(runPipeline(applyPm, moduleOp)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
