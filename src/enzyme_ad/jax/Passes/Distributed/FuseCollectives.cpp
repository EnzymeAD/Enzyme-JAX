#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveAtoms.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Utils.h"

#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_FUSECOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

using AtomPair = std::pair<AtomLabel, AtomLabel>;
using LabelKey = std::tuple<AtomSpace, size_t, size_t>;

LabelKey keyOf(const AtomLabel &label) {
  return {label.space, label.axis, label.atom};
}

// A chained pair: `second` consumes the result of `first` through
// `firstAwait`, which has no other user.
struct Chain {
  DistributedCollectiveOp first;
  DistributedAwait firstAwait;
  DistributedCollectiveOp second;
};

// The composition of two atomic collectives, on the atoms of their joint
// space: the fused collective's atom-to-atom pairs and reduced atoms.
struct Composition {
  SmallVector<AtomPair> pairs;
  SmallVector<AtomLabel> reduced;
};

// Everything needed to build the fused collective, computed without touching
// the IR.
struct FusionPlan {
  // Atom-level description of the fused collective, over the joint space's
  // atoms: one merged reduction group (if anything is reduced) and one pair
  // per atom, ready for buildAtomicOperands.
  CollectiveResolution fused;
  Composition composition;
  // The two collectives' own atom-level relations over the same atoms, kept
  // for the meaning-preservation check.
  SmallVector<AtomPair> firstPairs, secondPairs;
  SmallVector<AtomLabel> firstReduced;
  // The body the merged reduction group uses; null exactly when nothing is
  // reduced.
  Region *reductionBody = nullptr;
};

// Flattens a collective's mapping pairs to atom-to-atom pairs, in mapping
// order. Every pair's two sides have the same atom extents position by
// position (the joint refinement guarantees it).
SmallVector<AtomPair> flattenPairs(const CollectiveFactors &factors,
                                   const CollectiveAtoms &atoms) {
  SmallVector<AtomPair> result;
  for (const auto &[lhs, rhs] : factors.pairs) {
    SmallVector<AtomLabel> lhsLabels = atoms.labelsOf(lhs);
    SmallVector<AtomLabel> rhsLabels = atoms.labelsOf(rhs);
    for (auto [lhsLabel, rhsLabel] : llvm::zip_equal(lhsLabels, rhsLabels))
      if (atoms.extentOf(lhsLabel) != 1)
        result.push_back({lhsLabel, rhsLabel});
  }
  return result;
}

SmallVector<AtomLabel> flattenReduced(const CollectiveFactors &factors,
                                      const CollectiveAtoms &atoms) {
  SmallVector<AtomLabel> result;
  for (const ResolvedGroup &group : factors.reductionGroups)
    for (const AtomLabel &label : atoms.labelsOf(group))
      if (atoms.extentOf(label) != 1)
        result.push_back(label);
  return result;
}

// Composes `first` followed by `second`, both given as atom-level relations
// over one shared atom space in which first's output digits and second's
// input digits are the same labels (mesh atoms by identity, tile atoms
// through MidTile). Returns nullopt with `skipReason` set when the pair
// cannot be fused.
//
// Every digit that first produces (a pair's rhs) is one that second either
// consumes through a pair or reduces, except a replicate rhs: that is a
// discarded input digit, which stays a discarded input digit of the fused
// collective. Symmetrically a replicate lhs of second is a broadcast source
// that first knows nothing about and is carried over unchanged.
std::optional<Composition> composeCollectives(ArrayRef<AtomPair> firstPairs,
                                              ArrayRef<AtomLabel> firstReduced,
                                              ArrayRef<AtomPair> secondPairs,
                                              ArrayRef<AtomLabel> secondReduced,
                                              std::string &skipReason) {
  std::map<LabelKey, AtomLabel> secondConsumes;
  for (const auto &[lhs, rhs] : secondPairs)
    if (lhs.space != AtomSpace::Replicate)
      secondConsumes.emplace(keyOf(lhs), rhs);
  std::set<LabelKey> secondReducedKeys;
  for (const AtomLabel &label : secondReduced)
    secondReducedKeys.insert(keyOf(label));

  Composition result;
  result.reduced.append(firstReduced.begin(), firstReduced.end());
  // A replicate mapped to a replicate (a broadcast whose result is discarded,
  // as when a replicated value is sliced) relates no real digits.
  auto emit = [&](const AtomLabel &lhs, const AtomLabel &rhs) {
    if (lhs.space != AtomSpace::Replicate || rhs.space != AtomSpace::Replicate)
      result.pairs.push_back({lhs, rhs});
  };
  std::set<LabelKey> produced;
  for (const auto &[lhs, rhs] : firstPairs) {
    if (rhs.space == AtomSpace::Replicate) {
      emit(lhs, rhs);
      continue;
    }
    produced.insert(keyOf(rhs));
    if (auto it = secondConsumes.find(keyOf(rhs)); it != secondConsumes.end()) {
      emit(lhs, it->second);
      continue;
    }
    assert(secondReducedKeys.count(keyOf(rhs)) &&
           "a digit the first collective produces is neither consumed nor "
           "reduced by the second; distributed-make-replications-explicit "
           "must run first");
    // The second collective reduces this digit, so whatever the first
    // collective sent into it is what the fused collective reduces.
    assert(lhs.space != AtomSpace::Replicate &&
           "the second collective reduces an atom the first produced from a "
           "replicate; summing identical replicas is not meaningful");
    if (lhs.space == AtomSpace::InTile) {
      skipReason = "the second collective reduces over a device-local tile "
                   "atom of the first collective's input";
      return std::nullopt;
    }
    result.reduced.push_back(lhs);
  }
  for (const auto &[lhs, rhs] : secondPairs) {
    if (lhs.space == AtomSpace::Replicate) {
      emit(lhs, rhs);
      continue;
    }
    assert(produced.count(keyOf(lhs)) &&
           "the second collective consumes a digit the first never produces");
  }
  for (const AtomLabel &label : secondReduced) {
    assert(produced.count(keyOf(label)) &&
           "the second collective reduces a digit the first never produces");
    (void)label;
  }
  return result;
}

// Whether both collectives' reduction bodies agree on one recognized kind.
// On success `body` is one of the bodies (null when neither reduces).
bool checkReductionBodies(DistributedCollectiveOp first,
                          DistributedCollectiveOp second, Region *&body,
                          std::string &skipReason) {
  body = nullptr;
  std::optional<stablehlo::ReduceOpKind> kind;
  for (DistributedCollectiveOp collective : {first, second}) {
    for (Region &region : collective.getReductionBodies()) {
      stablehlo::ReduceOpKind bodyKind =
          stablehlo::classifyReduceBlockKind(region.front());
      if (kind && *kind != bodyKind) {
        skipReason = "the reduction bodies are different kinds";
        return false;
      }
      kind = bodyKind;
      if (!body)
        body = &region;
    }
  }
  if (kind && *kind == stablehlo::ReduceOpKind::Unknown) {
    skipReason = "the reduction body is not a recognized kind";
    return false;
  }
  return true;
}

ArrayRef<int64_t> tileShape(Type type) {
  return cast<RankedTensorType>(type).getShape();
}

std::set<size_t> meshAxesOf(const ResolvedGroup &factors) {
  std::set<size_t> axes;
  for (const ResolvedFactor &factor : factors)
    axes.insert(factor.key.second);
  return axes;
}

// A single-atom factor naming `label`, for buildAtomicOperands.
ResolvedFactor atomFactor(const CollectiveResolution &resolution,
                          const AtomLabel &label) {
  AxisKey key{label.space, label.axis};
  auto provenance = resolution.axisProvenance.find(key);
  return {key, resolution.atoms.extentOf(label),
          resolution.atoms.strideOf(label),
          provenance == resolution.axisProvenance.end()
              ? TypedValue<axis::AxisTypeInterface>()
              : provenance->second};
}

// Resolves the chain into one atom space and composes it. Returns nullopt
// with `skipReason` set when the pair cannot be fused; a collective that
// does not resolve at all is a bug upstream of this pass.
std::optional<FusionPlan>
planFusion(Chain chain, ArrayRef<PhysicalCommAxisType> meshAxisTypes,
           std::string &skipReason) {
  FusionPlan plan;
  Region *body;
  if (!checkReductionBodies(chain.first, chain.second, body, skipReason))
    return std::nullopt;
  plan.reductionBody = body;

  ArrayRef<int64_t> inputTile =
      tileShape(chain.first.getInputObject().getType());
  ArrayRef<int64_t> midTile =
      tileShape(chain.second.getInputObject().getType());
  ArrayRef<int64_t> outputTile = tileShape(chain.second.getOutputType());

  CollectiveResolution &fused = plan.fused;
  CollectiveAtoms &atoms = fused.atoms;
  for (size_t a = 0; a < meshAxisTypes.size(); ++a)
    atoms.addAxis({AtomSpace::Mesh, a}, meshAxisTypes[a].getExtent());
  for (size_t j = 0; j < inputTile.size(); ++j)
    atoms.addAxis({AtomSpace::InTile, j}, inputTile[j]);
  for (size_t j = 0; j < midTile.size(); ++j)
    atoms.addAxis({AtomSpace::MidTile, j}, midTile[j]);
  for (size_t j = 0; j < outputTile.size(); ++j)
    atoms.addAxis({AtomSpace::OutTile, j}, outputTile[j]);

  CollectiveResolutionError error;
  size_t nextReplicateId = 0;
  auto firstFactors = resolveCollectiveFactors(
      chain.first, meshAxisTypes, {AtomSpace::InTile, AtomSpace::MidTile},
      atoms, fused.axisProvenance, nextReplicateId, error);
  FailureOr<CollectiveFactors> secondFactors = failure();
  if (succeeded(firstFactors))
    secondFactors = resolveCollectiveFactors(
        chain.second, meshAxisTypes, {AtomSpace::MidTile, AtomSpace::OutTile},
        atoms, fused.axisProvenance, nextReplicateId, error);
  if (failed(firstFactors) || failed(secondFactors))
    llvm::report_fatal_error(
        llvm::Twine("distributed-fuse-collectives: resolveCollectiveFactors "
                    "hit a structural error on verifier-legal IR: ") +
        llvm::join(error.reasons, "; "));

  if (meshAxesOf(firstFactors->outputMeshFactors) !=
      meshAxesOf(secondFactors->inputMeshFactors)) {
    skipReason = "the collectives touch different mesh axes";
    return std::nullopt;
  }

  SmallVector<std::pair<ResolvedGroup, ResolvedGroup>> allPairs =
      firstFactors->pairs;
  allPairs.append(secondFactors->pairs.begin(), secondFactors->pairs.end());
  if (failed(atoms.refine(allPairs))) {
    skipReason = "the pair has no common atom basis";
    return std::nullopt;
  }

  plan.firstPairs = flattenPairs(*firstFactors, atoms);
  plan.firstReduced = flattenReduced(*firstFactors, atoms);
  plan.secondPairs = flattenPairs(*secondFactors, atoms);
  std::optional<Composition> composition =
      composeCollectives(plan.firstPairs, plan.firstReduced, plan.secondPairs,
                         flattenReduced(*secondFactors, atoms), skipReason);
  if (!composition)
    return std::nullopt;
  plan.composition = *composition;

  if (!composition->reduced.empty()) {
    ResolvedGroup group;
    for (const AtomLabel &label : composition->reduced)
      group.push_back(atomFactor(fused, label));
    fused.reductionGroups.push_back(std::move(group));
  } else {
    // checkReductionBodies only sees whether a reduction-group operand
    // exists, not whether atom refinement left it any live atoms (a group
    // can resolve to atoms that all turn out extent-1, e.g. a mesh axis that
    // refinement cuts down to nothing here); the body is only meaningful
    // when there is a reduced atom to attach it to.
    plan.reductionBody = nullptr;
  }
  for (const auto &[lhs, rhs] : composition->pairs)
    fused.pairs.push_back({{atomFactor(fused, lhs)}, {atomFactor(fused, rhs)}});
  assert((plan.reductionBody != nullptr) == !composition->reduced.empty() &&
         "a fused collective reduces exactly when one of the two did");
  return plan;
}

#ifndef NDEBUG
// An atom comparable across resolutions: replicate atoms have no identity
// beyond their extent (see AtomizeCollectives.cpp), every other atom is
// identified by where it sits on its axis.
using ComparableAtom = std::tuple<AtomSpace, size_t, size_t, uint64_t>;

ComparableAtom comparableAtom(const CollectiveAtoms &atoms,
                              const AtomLabel &label) {
  if (label.space == AtomSpace::Replicate)
    return {label.space, 0, 0, atoms.extentOf(label)};
  return {label.space, label.axis, label.atom, atoms.extentOf(label)};
}

// Standing meaning-preservation check. The rebuilt collective, resolved on
// its own, must be atomic and relate exactly the atoms the composition of the
// originals does; that catches errors in the rebuild. The composition itself
// is checked for conservation: every non-replicate digit the second
// collective writes is written once, and every non-replicate digit of the
// first collective's input is mapped or reduced once. Replicate digits are
// excluded because sources and sinks legitimately have no counterpart.
void assertFusionPreservesMeaning(
    const FusionPlan &plan, DistributedCollectiveOp fused,
    ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  CollectiveResolutionError error;
  FailureOr<CollectiveResolution> post = resolveCollectiveAtoms(
      fused, meshAxisTypes, tileShape(fused.getInputObject().getType()),
      tileShape(fused.getOutputType()), error);
  assert(succeeded(post) &&
         "fuse-collectives: a just-fused collective must resolve to common "
         "atoms");
  assert(isCollectiveAtomic(*post) &&
         "fuse-collectives: the fused collective must be atomic");

  SmallVector<std::pair<ComparableAtom, ComparableAtom>> expected, actual;
  for (const auto &[lhs, rhs] : plan.composition.pairs)
    expected.push_back({comparableAtom(plan.fused.atoms, lhs),
                        comparableAtom(plan.fused.atoms, rhs)});
  for (const auto &[lhsGroup, rhsGroup] : post->pairs) {
    SmallVector<AtomLabel> lhsLabels = post->atoms.labelsOf(lhsGroup);
    SmallVector<AtomLabel> rhsLabels = post->atoms.labelsOf(rhsGroup);
    for (auto [lhs, rhs] : llvm::zip_equal(lhsLabels, rhsLabels))
      actual.push_back(
          {comparableAtom(post->atoms, lhs), comparableAtom(post->atoms, rhs)});
  }
  assert(expected == actual &&
         "fuse-collectives: the fused collective's pairs are not the "
         "composition of the originals");

  using Digits = std::multiset<ComparableAtom>;
  auto realDigit = [](const ComparableAtom &atom) {
    return std::get<0>(atom) != AtomSpace::Replicate;
  };
  Digits outputs, inputs, expectedOutputs, expectedInputs;
  auto insertReal = [&](Digits &set, const AtomLabel &label) {
    ComparableAtom atom = comparableAtom(plan.fused.atoms, label);
    if (realDigit(atom))
      set.insert(atom);
  };
  for (const auto &[lhs, rhs] : plan.composition.pairs) {
    insertReal(inputs, lhs);
    insertReal(outputs, rhs);
  }
  for (const AtomLabel &label : plan.composition.reduced)
    insertReal(inputs, label);
  for (const auto &[lhs, rhs] : plan.secondPairs)
    insertReal(expectedOutputs, rhs);
  for (const auto &[lhs, rhs] : plan.firstPairs)
    insertReal(expectedInputs, lhs);
  for (const AtomLabel &label : plan.firstReduced)
    insertReal(expectedInputs, label);
  assert(outputs == expectedOutputs &&
         "fuse-collectives: fusion lost or duplicated an output digit");
  assert(inputs == expectedInputs &&
         "fuse-collectives: fusion lost or duplicated an input digit");
}
#endif

// Replaces `chain` by one collective built from `plan`, which reads the first
// collective's input and yields the second's output.
void applyFusion(Chain chain, const FusionPlan &plan,
                 ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  DistributedCollectiveOp second = chain.second;
  assert(llvm::hasSingleElement(second->getUsers()) &&
         "a DistributedCollective's async handle must have exactly one "
         "DistributedAwait consumer (see createCollectiveAndAwait)");
  auto secondAwait = cast<DistributedAwait>(*second->getUsers().begin());

  OpBuilder builder(second);
  Location loc = builder.getFusedLoc({chain.first.getLoc(), second.getLoc()});
  AtomicOperands operands =
      buildAtomicOperands(plan.fused, meshAxisTypes, builder, loc);
  CollectiveAndAwait created = createCollectiveAndAwait(
      builder, loc, chain.first.getInputObject(), operands.inputMesh,
      operands.outputMesh, operands.reductionGroups, operands.mapping,
      second.getOutputType());
  if (plan.reductionBody) {
    IRMapping mapper;
    plan.reductionBody->cloneInto(&created.collective.getReductionBodies()[0],
                                  mapper);
  }

#ifndef NDEBUG
  assertFusionPreservesMeaning(plan, created.collective, meshAxisTypes);
#endif

  secondAwait.getValue().replaceAllUsesWith(created.await.getValue());
  secondAwait->erase();
  second->erase();
  chain.firstAwait->erase();
  chain.first->erase();
}

struct FuseCollectivesPass
    : public impl::FuseCollectivesPassBase<FuseCollectivesPass> {
  using FuseCollectivesPassBase::FuseCollectivesPassBase;

  // Finds the collective whose result feeds `second`, if any. Sets
  // `skipReason` when there is one but it cannot be fused for a structural
  // reason (its result has other users).
  // Looks for the Await directly, not through any intervening cast: by this
  // point in the pipeline distributed-merge-adjacent-trivial-kernels has
  // already required boundary casts to be gone, so nothing sits between an
  // Await and the next collective's input.
  static std::optional<Chain> findChain(DistributedCollectiveOp second,
                                        std::string &skipReason) {
    auto await = second.getInputObject().getDefiningOp<DistributedAwait>();
    if (!await)
      return std::nullopt;
    auto first =
        await.getAsyncHandle().getDefiningOp<DistributedCollectiveOp>();
    if (!first)
      return std::nullopt;
    if (!await.getValue().hasOneUse()) {
      skipReason = "the first collective's result has other users";
      return std::nullopt;
    }
    return Chain{first, await, second};
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    if (failed(physicalMesh)) {
      // findUniquePhysicalMesh already emitted a diagnostic.
      signalPassFailure();
      return;
    }
    SmallVector<PhysicalCommAxisType> meshAxisTypes;
    for (Attribute axisAttr : physicalMesh->getAxesAttr())
      meshAxisTypes.push_back(
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));

    // Each sweep fuses at most one pair and then restarts, so a fused
    // collective is reconsidered against both neighbours and chains collapse.
    // The remarks come from the last sweep only, where nothing fuses and so
    // every reported skip is final: a skip reason recomputed mid-chain (e.g.
    // "other users" on a collective a later fusion will remove) would
    // otherwise be stale by the time the module settles. A single greedy
    // rewrite pattern can't express that "only report once nothing more will
    // change" condition, which is why this pass re-walks the module instead
    // of using applyPatternsGreedily as its sibling cleanup passes do; module
    // sizes here (dozens of collectives) make the O(n^2) worst case cheap.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<DistributedCollectiveOp> collectives;
      module.walk([&](DistributedCollectiveOp c) { collectives.push_back(c); });

      SmallVector<std::pair<DistributedCollectiveOp, std::string>> skipped;
      for (DistributedCollectiveOp second : collectives) {
        std::string skipReason;
        std::optional<Chain> chain = findChain(second, skipReason);
        std::optional<FusionPlan> plan;
        if (chain)
          plan = planFusion(*chain, meshAxisTypes, skipReason);
        if (!plan) {
          if (!skipReason.empty())
            skipped.push_back({second, skipReason});
          continue;
        }

        applyFusion(*chain, *plan, meshAxisTypes);
        changed = true;
        break;
      }
      if (!changed)
        for (auto &[second, reason] : skipped)
          second->emitRemark()
              << "fuse-collectives: not fused with the preceding collective ("
              << reason << ")";
    }

    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
