#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveAtoms.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_ATOMIZECOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

using TV_AxisFactor = TypedValue<axis::AxisFactorType>;

// Prints one atom as `<space><axis>.<atom>` (e.g. `mesh0.1`, `in2.0`): a
// handle for a lit test (or a future NormalizedCollective builder) to key
// off of that only depends on the resolution's own structure, never on an
// SSA value name.
void printAtomLabel(llvm::raw_ostream &os, const AtomLabel &label) {
  switch (label.space) {
  case AtomSpace::Mesh:
    os << "mesh";
    break;
  case AtomSpace::InTile:
    os << "in";
    break;
  case AtomSpace::OutTile:
    os << "out";
    break;
  case AtomSpace::Replicate:
    os << "replicate";
    break;
  }
  os << label.axis << "." << label.atom;
}

void printAtomLabels(llvm::raw_ostream &os, ArrayRef<AtomLabel> labels) {
  llvm::interleaveComma(
      labels, os, [&](const AtomLabel &label) { printAtomLabel(os, label); });
}

// Renders a resolved collective as one line per mesh axis and per tile
// dimension (its atoms' extents, major-first) followed by one line per
// reduction group and per mapping pair (the atoms it covers, as `pair k: lhs
// -> rhs`). This is deliberately the same information a NormalizedCollective
// builder (a later pass) will need to rebuild the atomic form, so pinning
// this text in a lit test doubles as a check on the resolution itself, not
// just on this pass's own formatting.
std::string describeResolution(ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                               ArrayRef<int64_t> inputTile,
                               ArrayRef<int64_t> outputTile,
                               const CollectiveResolution &resolution) {
  std::string message;
  llvm::raw_string_ostream os(message);
  const CollectiveAtoms &atoms = resolution.atoms;

  auto printAxisAtoms = [&](StringRef prefix, AtomSpace space, size_t axis) {
    os << prefix << axis << ": ";
    llvm::interleaveComma(
        atoms.labelsOfAxis({space, axis}), os,
        [&](const AtomLabel &label) { os << atoms.extentOf(label); });
    os << "\n";
  };
  for (size_t a = 0; a < meshAxisTypes.size(); ++a)
    printAxisAtoms("mesh", AtomSpace::Mesh, a);
  for (size_t d = 0; d < inputTile.size(); ++d)
    printAxisAtoms("in", AtomSpace::InTile, d);
  for (size_t d = 0; d < outputTile.size(); ++d)
    printAxisAtoms("out", AtomSpace::OutTile, d);

  for (auto [i, group] : llvm::enumerate(resolution.reductionGroups)) {
    os << "reduction" << i << ": ";
    printAtomLabels(os, atoms.labelsOf(group));
    os << "\n";
  }
  for (auto [i, pair] : llvm::enumerate(resolution.pairs)) {
    os << "pair" << i << ": ";
    printAtomLabels(os, atoms.labelsOf(pair.first));
    os << " -> ";
    printAtomLabels(os, atoms.labelsOf(pair.second));
    os << "\n";
  }
  return message;
}

// An atom, made comparable across two independently-computed resolutions.
// Mesh/tile atoms compare by (space, axis, atom index): meaningful identity,
// since a mesh axis's index comes from the module's fixed physical mesh and
// a tile axis's from the tensor's own rank, independent of which pair or
// group happens to reference it. A replicate atom has no such identity --
// resolveCollectiveAtoms hands out a fresh one-off axis id per FACTOR
// occurrence (not per underlying ReplicationAxis value), so splitting one
// coarser pair into several atomic ones inherently reallocates those ids;
// only its extent is meaningful to compare.
struct ComparableAtom {
  AtomSpace space;
  size_t axis;
  size_t atom;
  uint64_t extent;

  bool operator==(const ComparableAtom &other) const {
    if (space != other.space)
      return false;
    if (space == AtomSpace::Replicate)
      return extent == other.extent;
    return axis == other.axis && atom == other.atom;
  }
};

ComparableAtom comparableAtom(const CollectiveResolution &resolution,
                              const AtomLabel &label) {
  return {label.space, label.axis, label.atom,
          resolution.atoms.extentOf(label)};
}

// Whether `before` and `after` cover the same atoms, related the same way,
// ignoring how many mapping-pair or reduction-group *operands* each one
// happens to bucket them into. Atomizing only ever refines that bucketing
// (splitting a pair or group that spans several atoms into one-atom pieces);
// it never changes which atom exists or which atom relates to which, so this
// is the right notion of "unchanged meaning" for the rewrite's self-check --
// stricter than comparing extents alone, but looser than exact textual
// equality of describeResolution's own pair/group line boundaries, since the
// rewrite is expected to move those boundaries.
bool haveEquivalentAtomStructure(ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                 ArrayRef<int64_t> inputTile,
                                 ArrayRef<int64_t> outputTile,
                                 const CollectiveResolution &before,
                                 const CollectiveResolution &after) {
  auto axisExtents = [](const CollectiveResolution &r, AtomSpace space,
                        size_t axis) {
    SmallVector<uint64_t> extents;
    for (AtomLabel label : r.atoms.labelsOfAxis({space, axis}))
      extents.push_back(r.atoms.extentOf(label));
    return extents;
  };
  for (size_t a = 0; a < meshAxisTypes.size(); ++a)
    if (axisExtents(before, AtomSpace::Mesh, a) !=
        axisExtents(after, AtomSpace::Mesh, a))
      return false;
  for (size_t d = 0; d < inputTile.size(); ++d)
    if (axisExtents(before, AtomSpace::InTile, d) !=
        axisExtents(after, AtomSpace::InTile, d))
      return false;
  for (size_t d = 0; d < outputTile.size(); ++d)
    if (axisExtents(before, AtomSpace::OutTile, d) !=
        axisExtents(after, AtomSpace::OutTile, d))
      return false;

  // Flattened across every group/pair, dropping which original operand each
  // atom came from -- exactly the bucketing the rewrite is allowed to move.
  auto flattenedReductionAtoms = [](const CollectiveResolution &r) {
    SmallVector<ComparableAtom> atoms;
    for (const ResolvedGroup &group : r.reductionGroups)
      for (const AtomLabel &label : r.atoms.labelsOf(group))
        atoms.push_back(comparableAtom(r, label));
    return atoms;
  };
  if (flattenedReductionAtoms(before) != flattenedReductionAtoms(after))
    return false;

  auto flattenedPairAtoms = [](const CollectiveResolution &r) {
    SmallVector<std::pair<ComparableAtom, ComparableAtom>> pairs;
    for (const auto &[lhsGroup, rhsGroup] : r.pairs) {
      SmallVector<AtomLabel> lhsLabels = r.atoms.labelsOf(lhsGroup);
      SmallVector<AtomLabel> rhsLabels = r.atoms.labelsOf(rhsGroup);
      for (auto [lhsLabel, rhsLabel] : llvm::zip_equal(lhsLabels, rhsLabels))
        pairs.emplace_back(comparableAtom(r, lhsLabel),
                           comparableAtom(r, rhsLabel));
    }
    return pairs;
  };
  return flattenedPairAtoms(before) == flattenedPairAtoms(after);
}

// Builds a fresh axis.factor over one atom, at module scope: the digit the
// atom names is, by construction, exactly the extent/stride absolute cut
// `atoms` computed for it, over the axis its originating factor came from.
// Skipped by every caller for an extent-1 atom, which carries no data and
// would otherwise need a provenance value even for tile axes no factor in
// the collective ever actually references.
//
// A replicate atom is handled differently from a mesh/tile one: a replicate
// axis has no identity worth preserving across factors in the first place
// (resolveCollectiveAtoms already treats every replicate factor as its own
// one-off axis slot, sized to that factor's own extent -- see its "Replicate
// axes are numbered in resolution order" comment), so a strided sub-factor
// of the original, possibly coarser replicate axis would misrepresent this
// atom's size the moment anything re-resolves it. A fresh, exactly-sized
// ReplicationAxis stands in for it instead, the same "just need something
// with the same extent" idiom distributed-make-replications-explicit uses.
TV_AxisFactor buildAtomFactor(const CollectiveResolution &resolution,
                              const AtomLabel &label, OpBuilder &builder,
                              Location loc) {
  auto extent = static_cast<int32_t>(resolution.atoms.extentOf(label));
  if (label.space == AtomSpace::Replicate) {
    auto replicationAxis = builder.create<ReplicationAxisOp>(loc, extent);
    return axis::viewAxisAsFactor(replicationAxis.getAxis(), builder, loc);
  }
  TypedValue<axis::AxisTypeInterface> provenance =
      resolution.axisProvenance.at({label.space, label.axis});
  return builder.create<axis::AxisFactorOp>(
      loc, provenance, extent,
      static_cast<int32_t>(resolution.atoms.strideOf(label)));
}

// Builds one fresh factor per atom in `labels` whose extent is greater than
// one, in order.
SmallVector<TV_AxisFactor>
buildAtomFactors(const CollectiveResolution &resolution,
                 ArrayRef<AtomLabel> labels, OpBuilder &builder, Location loc) {
  SmallVector<TV_AxisFactor> factors;
  for (const AtomLabel &label : labels)
    if (resolution.atoms.extentOf(label) != 1)
      factors.push_back(buildAtomFactor(resolution, label, builder, loc));
  return factors;
}

// Rewrites `collective`'s own input_mesh/output_mesh/reduction_groups/
// mapping operands into atomic form: one factor per atom, with every mesh
// atom appearing once on each mesh operand and every mapping pair relating
// exactly one factor to one factor. This re-expresses the collective's
// existing index-space coverage on a finer, common basis; it never changes
// what the collective computes, only how finely its own operands cut the
// axes they already cut. `resolution` must be `collective`'s own resolution,
// from a call already made by the caller (resolveCollectiveAtoms is not
// cheap enough to justify calling it twice for the same collective).
//
// Every fresh factor/product/map is built independently at each use site,
// even where two sites end up describing the same atom (e.g. input_mesh and
// output_mesh cover the same physical atoms once
// distributed-make-replications-explicit has run): a later cse handles that
// deduplication, and this rewrite never mutates or reuses an existing
// axis-algebra value, since `mapping` (and, in principle, a mesh operand)
// may be shared with another collective that must keep seeing its own
// original operands unchanged.
void atomizeCollective(DistributedCollectiveOp collective,
                       const CollectiveResolution &resolution,
                       ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  OpBuilder builder(collective);
  Location loc = collective.getLoc();
  const CollectiveAtoms &atoms = resolution.atoms;

  Value newInputMesh, newOutputMesh, newMap;
  SmallVector<Value> newReductionGroups;
  {
    // Every op built in this block is pure axis-algebra metadata and
    // belongs at module scope; only the final operand reassignment below
    // needs to happen at the collective's own position.
    axis::ModuleScopeGuard moduleScope(builder);

    // input_mesh/output_mesh: every mesh atom, axis 0's atoms then axis 1's,
    // and so on, assembled into one product. Built twice (once per side)
    // rather than once and reused, since the two operands are logically
    // independent even when they end up structurally identical.
    auto buildMeshOperand = [&] {
      SmallVector<TV_AxisFactor> factors;
      for (size_t a = 0; a < meshAxisTypes.size(); ++a)
        factors.append(buildAtomFactors(
            resolution, atoms.labelsOfAxis({AtomSpace::Mesh, a}), builder,
            loc));
      return Value(axis::viewFactorsAsProduct(factors, builder, loc));
    };
    newInputMesh = buildMeshOperand();
    newOutputMesh = buildMeshOperand();

    // Each reduction group's original factors, atomized in place and
    // reassembled into one product per group; the reduction body region
    // (kept as-is) and the operand's position in reduction_groups are
    // untouched.
    for (const ResolvedGroup &group : resolution.reductionGroups) {
      SmallVector<TV_AxisFactor> factors;
      for (const ResolvedFactor &factor : group)
        factors.append(
            buildAtomFactors(resolution, atoms.labelsOf(factor), builder, loc));
      newReductionGroups.push_back(
          axis::viewFactorsAsProduct(factors, builder, loc));
    }

    // Mapping: one atomic (single-factor) pair per atom position of every
    // original pair, all collected into ONE fresh axis.map -- never one
    // axis.map per original pair. A pair's lhs/rhs atom runs are positionally
    // aligned one-to-one by computeCommonAtomsAndMappingSplits (the same
    // property describeResolution's own pairK: lhs -> rhs line relies on), so
    // the two extents at a given position always agree and either both or
    // neither side is skipped there.
    SmallVector<Value> newMappingLhs, newMappingRhs;
    for (const auto &[lhsGroup, rhsGroup] : resolution.pairs) {
      SmallVector<AtomLabel> lhsLabels = atoms.labelsOf(lhsGroup);
      SmallVector<AtomLabel> rhsLabels = atoms.labelsOf(rhsGroup);
      for (auto [lhsLabel, rhsLabel] : llvm::zip_equal(lhsLabels, rhsLabels)) {
        if (atoms.extentOf(lhsLabel) == 1)
          continue;
        TV_AxisFactor lhsFactor =
            buildAtomFactor(resolution, lhsLabel, builder, loc);
        TV_AxisFactor rhsFactor =
            buildAtomFactor(resolution, rhsLabel, builder, loc);
        newMappingLhs.push_back(
            axis::viewFactorsAsProduct(lhsFactor, builder, loc));
        newMappingRhs.push_back(
            axis::viewFactorsAsProduct(rhsFactor, builder, loc));
      }
    }
    newMap = builder
                 .create<axis::AxisMapOp>(loc, ValueRange(newMappingLhs),
                                          ValueRange(newMappingRhs))
                 .getMap();
  }

  collective.getInputMeshMutable().assign(newInputMesh);
  collective.getOutputMeshMutable().assign(newOutputMesh);
  collective.getReductionGroupsMutable().assign(newReductionGroups);
  collective.getMappingMutable().assign(newMap);
}

struct AtomizeCollectivesPass
    : public impl::AtomizeCollectivesPassBase<AtomizeCollectivesPass> {
  using AtomizeCollectivesPassBase::AtomizeCollectivesPassBase;

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

    module.walk([&](DistributedCollectiveOp collective) {
      // A DistributedCollective's async handle always has exactly one
      // DistributedAwait consumer (see createCollectiveAndAwait); that
      // Await's result is the output tile shape (no mesh prefix, same as
      // input_object), and is the dialect-idiomatic way to reach it rather
      // than re-deriving it from output_type.
      assert(llvm::hasSingleElement(collective->getUsers()) &&
             "a DistributedCollective's async handle must have exactly one "
             "DistributedAwait consumer (see createCollectiveAndAwait)");
      auto await = cast<DistributedAwait>(*collective->getUsers().begin());

      auto inputTile =
          cast<RankedTensorType>(collective.getInputObject().getType())
              .getShape();
      auto outputTile =
          cast<RankedTensorType>(await.getValue().getType()).getShape();

      CollectiveResolutionError error;
      FailureOr<CollectiveResolution> resolution = resolveCollectiveAtoms(
          collective, meshAxisTypes, inputTile, outputTile, error);
      if (failed(resolution)) {
        if (error.kind != CollectiveResolutionError::Kind::NoCommonAtoms) {
          // Structural errors (a factor group not built from axis.product, a
          // factor with no provenance axis, or a physical axis outside the
          // module's mesh) are exactly what the op verifier and this pass's
          // own precondition -- running after
          // distributed-make-replications-explicit, against the same single
          // physical mesh -- already rule out for IR that reaches this pass.
          llvm::report_fatal_error(
              llvm::Twine("distributed-atomize-collectives: "
                          "resolveCollectiveAtoms hit a structural error on "
                          "verifier-legal IR: ") +
              llvm::join(error.reasons, "; "));
        }
        // An indivisible mapping pair or non-nesting cuts: a legitimate
        // infeasible sharding candidate, not a bug (see this pass's own
        // description in Passes.td).
        collective->emitRemark()
            << "atomize-collectives: could not split into common atoms ("
            << error.reasons.front() << ")";
        return;
      }

      atomizeCollective(collective, *resolution, meshAxisTypes);

      // Re-resolve the now-rewritten collective rather than trusting that
      // `resolution` still describes it: the remark is a standing debugging
      // aid that should describe the collective as it now stands, and
      // re-resolving it for real (not just reusing `resolution`) turns
      // printing it into a strong self-check on the rewrite itself --
      // input_object/output_type are untouched by the rewrite, so
      // inputTile/outputTile are still valid for it.
      CollectiveResolutionError postError;
      FailureOr<CollectiveResolution> postResolution = resolveCollectiveAtoms(
          collective, meshAxisTypes, inputTile, outputTile, postError);
      assert(succeeded(postResolution) &&
             "atomize-collectives: a just-atomized collective must still "
             "resolve to common atoms");
      assert(haveEquivalentAtomStructure(meshAxisTypes, inputTile, outputTile,
                                         *resolution, *postResolution) &&
             "atomize-collectives: rewriting to atomic form changed the "
             "collective's own atoms or how they relate to each other");
      assert(isCollectiveAtomic(*postResolution) &&
             "atomize-collectives: rewriting to atomic form did not actually "
             "produce an atomic collective");
      collective->emitRemark() << describeResolution(
          meshAxisTypes, inputTile, outputTile, *postResolution);
    });

    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
