#include "src/enzyme_ad/jax/Passes/Distributed/BeamSearchDriver.h"
#include "src/enzyme_ad/jax/Passes/Distributed/LogicalAxisOverlap.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ReplayTree.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"

#include <limits>
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
  int extentRemaining;

  // Factors of the physical mesh available for this axis
  std::vector<SharedOpRef<AxisFactorOp>> availableSpace;

  StrategySearchNode() = delete; // disable default constructor

  StrategySearchNode(const StrategySearchNode &other,
                     std::shared_ptr<AxisMappingReplayTree> childDecisions)
      : axes(other.axes), axisIndex(other.axisIndex), decisions(childDecisions),
        extentRemaining(other.extentRemaining),
        availableSpace(other.availableSpace) {}

public:
  StrategySearchNode(std::shared_ptr<const LogicalAxisOrder> axes,
                     std::shared_ptr<AxisMappingReplayTree> decisions)
      : axes(axes), axisIndex(-1 /* incremented to 0 upon setupNextAxis */),
        decisions(decisions), availableSpace(), extentRemaining(-1) {}

  TypedValue<LogicalMeshAxisType> currentAxis() const {
    assert(!finalized());
    return (*axes)[axisIndex];
  }
  bool finalized() const override { return axisIndex >= axes->size(); }
  int getExtentRemaining() const { return extentRemaining; }
  const auto &getAvailableSpace() const { return availableSpace; }

  std::shared_ptr<StrategySearchNode> makeChild() const {
    auto childDecisions = AxisMappingReplayTree::makeChild(decisions);
    auto childNode = std::shared_ptr<StrategySearchNode>(
        new StrategySearchNode(*this, childDecisions));
    return childNode;
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
    int factorExtent = getFactorExtent(factorValue);
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
 * This pass expects to be given a cloned module from the one the
 * decisions were made for. As such, an IR mapper is required to go
 * from the original to logical axes. We also expect each logical
 * axis to currently not be factored: any factors should have the
 * whole extent of the axis. This pass effectively replaces any
 * products of the axis factors with ones using the decisions from the search
 * node.
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
  ApplyPartialDecisions(std::shared_ptr<StrategySearchNode> node,
                        IRMapping &originalToCloned)
      : node(node), originalToCloned(&originalToCloned) {}

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
        new ApplyPartialDecisions(node, originalToCloned));
  }

  void runOnOperation() override {
    ModuleOp clonedModule = getOperation();
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(clonedModule.getBody());

    for (auto &[origAxis, origFactors] : node->getPartialDecisions()) {
      if (origFactors.empty()) {
        // Nothing decided for this axis yet; leave its pristine
        // representation untouched.
        continue;
      }

      Value clonedAxisVal = originalToCloned->lookupOrNull(origAxis);
      if (!clonedAxisVal) {
        origAxis.getDefiningOp()->emitError(
            "logical axis has no counterpart in the cloned module");
        return signalPassFailure();
      }
      auto clonedAxis = cast<TypedValue<AxisTypeInterface>>(clonedAxisVal);

      // Discover what to replace *before* building any new axis.factor
      // referencing clonedAxis -- otherwise a newly-built factor (e.g. the
      // residual below) would itself show up as a use of clonedAxis and be
      // mistaken for part of its current state.
      auto oldFactorOps = findAxisFactors(clonedAxis);
      if (failed(oldFactorOps))
        return signalPassFailure();
      llvm::SmallVector<TypedValue<AxisFactorType>> oldFactors;
      for (AxisFactorOp oldFactorOp : *oldFactorOps)
        oldFactors.push_back(castTypedValue<AxisFactorType>(
            oldFactorOp.getResult(), "AxisFactorType"));

      llvm::SmallVector<TypedValue<AxisFactorType>> clonedFactors;
      int decidedExtent = 1;
      for (TypedValue<AxisFactorType> origFactor : origFactors) {
        auto origProvenance = getFactorProvenanceAxis(origFactor);
        if (failed(origProvenance)) {
          origFactor.getDefiningOp()->emitError(
              "failed to resolve provenance axis for a decided factor");
          return signalPassFailure();
        }
        Value clonedProvenance = originalToCloned->lookupOrNull(*origProvenance);
        if (!clonedProvenance) {
          // A search-created axis (e.g. a device-local serialization axis)
          // with no counterpart in the original module: materialize it here
          // and register the mapping for reuse.
          builder.clone(*origProvenance->getDefiningOp(), *originalToCloned);
          clonedProvenance = originalToCloned->lookup(*origProvenance);
        }
        int extent = getFactorExtent(origFactor);
        decidedExtent *= extent;
        auto clonedFactor = builder.create<AxisFactorOp>(
            origFactor.getLoc(), clonedProvenance, extent,
            getFactorStride(origFactor));
        clonedFactors.push_back(
            castTypedValue<AxisFactorType>(clonedFactor.getResult(),
                                           "AxisFactorType"));
      }

      int remainder = getAxisExtent(clonedAxis) / decidedExtent;
      if (remainder > 1) {
        auto residualFactor = builder.create<AxisFactorOp>(
            clonedAxisVal.getLoc(), clonedAxisVal, remainder, 1);
        clonedFactors.push_back(
            castTypedValue<AxisFactorType>(residualFactor.getResult(),
                                           "AxisFactorType"));
      }

      if (failed(replaceAxisFactors(oldFactors, clonedFactors, builder)))
        return signalPassFailure();
    }
  }
};

class StrategyExplorer : public BeamSearchExplorerBase<StrategySearchNode> {
public:
  LogicalAxisOverlap &overlap;
  OpBuilder &builder;
  ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace;
  Location defaultLoc;

  StrategyExplorer(LogicalAxisOverlap &overlap, OpBuilder &builder,
                   ArrayRef<SharedOpRef<AxisFactorOp>> totalMeshSpace,
                   Location defaultLoc)
      : BeamSearchExplorerBase<StrategySearchNode>(), overlap(overlap),
        builder(builder), totalMeshSpace(totalMeshSpace),
        defaultLoc(defaultLoc) {}

  // Not algorithmically fast! But we expect resonably small (10000k max,
  // maybe) and easily divisible numbers.
  llvm::SmallVector<uint> uniquePrimeFactors(int n) {
    llvm::SmallVector<uint> primes;
    int thresh = 1;
    int i = 2;
    while (i <= n) {
      if (n % i == 0) {
        if (i > thresh) {
          primes.push_back(i);
          thresh = i;
        }
        n /= i;
        i = 2;
      } else {
        i++;
      }
    }
    return primes;
  }

  virtual std::vector<std::shared_ptr<StrategySearchNode>>
  generateCandidatesFromNode(
      std::shared_ptr<StrategySearchNode> node) override {
    // For the axis being worked on, decide if we want to
    // apply a sharding axis, pipeline (TODO), or place within
    // a device.
    const int extentRemaining = node->getExtentRemaining();

    std::vector<std::shared_ptr<StrategySearchNode>> candidates;

    // Decision type 1: for factor of the physical space, see if we
    // can pull out a prime factor
    {
      auto possibleChunks = uniquePrimeFactors(extentRemaining);
      auto availableFactors = node->getAvailableSpace();
      for (auto physicalFactor : availableFactors) {
        // TODO do we have to swap available extent to owning things too?
        int factorExtent = getFactorExtent(physicalFactor->get());
        int chunkTaken = -1;
        for (auto chunk : possibleChunks) {
          if (chunk <= factorExtent && factorExtent % chunk == 0) {
            chunkTaken = chunk;
            break;
          }
        }
        if (chunkTaken == -1)
          continue;

        int residualExtent = factorExtent / chunkTaken;

        // We have a candidate that takes a chunk out of the physical factor,
        // from the high stride positions. Need to take the chunk factor,
        // and create a residual factor from the remaining extent.
        auto takenFactor =
            sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(
                createSubfactor(physicalFactor->get(), chunkTaken,
                                residualExtent, builder,
                                physicalFactor->get().getLoc()));
        auto residualFactor =
            sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(
                createSubfactor(physicalFactor->get(), residualExtent, 1,
                                builder, physicalFactor->get().getLoc()));
        auto child = node->makeChild();
        child->addFactor(takenFactor);
        child->removeAvailableSpace(physicalFactor);
        child->addAvailableSpace(residualFactor);
        child->considerNextAxis(overlap, totalMeshSpace, builder);
        candidates.push_back(child);
      }
    }

    // Decision type 2: serialize remaining
    {
      auto child = node->makeChild();
      // spin up a fresh new owned reference to a within-device
      // axis, factor with extent equal to remaining
      DeviceLocalAxisOp localizationAxis =
          builder.create<DeviceLocalAxisOp>(defaultLoc, extentRemaining);
      auto factor = viewAxisAsFactor(localizationAxis, builder, defaultLoc);
      SharedOpRef<AxisFactorOp> owningFactor =
          sharedOpRefFromValue<TypedValue<AxisFactorType>, AxisFactorOp>(
              factor);
      SharedOpRef<Operation *> owningAxis =
          std::make_shared<OwningOpRef<Operation *>>(localizationAxis);
      child->addFactor(owningFactor);
      child->addSupportingIR(owningAxis);
      child->considerNextAxis(overlap, totalMeshSpace, builder);
      candidates.push_back(child);
    }
    return candidates;
  }
};

class StrategyScorer : public BeamSearchScorerBase<StrategySearchNode> {
  ModuleOp originalModule;
  bool dumpCandidates;

public:
  StrategyScorer(ModuleOp originalModule, bool dumpCandidates)
      : originalModule(originalModule), dumpCandidates(dumpCandidates) {}

  // Plan: run a pass pipeline to apply and lower the current decisions
  // and score the result. Pipeline:
  // - ApplyPartialDecisions : decisions from the search
  // - ApplyHeuristicDecisions : heuristically complete the remaining
  //   decisions
  // - Lowering pipeline : general lowering pipeline we will import from
  //   outside this pass
  // - ScoreModel : evaluate the lowered IR to produce a score
  //
  // Only ApplyPartialDecisions is implemented so far; the rest remain TODO,
  // so this returns a placeholder score (0.0 on success, -infinity if
  // ApplyPartialDecisions fails on the candidate's decisions).
  //
  // The candidate is scored on a clone of the original module, built fresh
  // per call via a standalone PassManager (not Pass::runPipeline, which
  // requires its target to be nested under the operation the calling pass is
  // currently processing -- our clone is a disconnected top-level module).
  double score(const std::shared_ptr<StrategySearchNode> &node) override {
    IRMapping mapper;
    OwningOpRef<ModuleOp> clonedModule(
        cast<ModuleOp>(originalModule->clone(mapper)));

    PassManager pm(originalModule.getContext(), ModuleOp::getOperationName());
    pm.addPass(ApplyPartialDecisions::create(node, mapper));
    LogicalResult pipelineResult = pm.run(*clonedModule);

    if (dumpCandidates) {
      llvm::errs() << "// ApplyPartialDecisions candidate ("
                   << (succeeded(pipelineResult) ? "ok" : "FAILED") << "):\n";
      clonedModule->print(llvm::errs());
      llvm::errs() << "\n";
    }

    if (failed(pipelineResult))
      return -std::numeric_limits<double>::infinity();

    return 0.0; // TODO: ApplyHeuristicDecisions + lowering + ScoreModel
  }
};

struct DistributedSearchStrategiesPass
    : public impl::DistributedSearchStrategiesPassBase<
          DistributedSearchStrategiesPass> {
  using DistributedSearchStrategiesPassBase::DistributedSearchStrategiesPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    mlir::OpBuilder builder(moduleOp.getContext());
    builder.clearInsertionPoint();

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

    int TODO_PARAMETER_BEAM_SIZE = 100;
    BeamSearchBreadthFirstQueue<StrategySearchNode> queue(
        TODO_PARAMETER_BEAM_SIZE);
    queue.push(initialNode);
    StrategyExplorer explorer(overlap, builder, physicalAxes,
                              moduleOp.getLoc());
    StrategyScorer scorer(moduleOp, dumpCandidates);

    BeamSearchDriver<StrategySearchNode> driver(queue, scorer, explorer);
    driver.run();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
