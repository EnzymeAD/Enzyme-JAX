#include "CollectiveAtoms.h"

#include "llvm/ADT/STLExtras.h"

#include <set>

namespace mlir::enzyme::distributed {

namespace {
using TV_AxisFactor = TypedValue<axis::AxisFactorType>;
using TV_FactorGroup = TypedValue<axis::FactorGroupType>;
} // namespace

FailureOr<CollectiveFactors> resolveCollectiveFactors(
    DistributedCollectiveOp collective,
    ArrayRef<PhysicalCommAxisType> meshAxisTypes, TileSpaces spaces,
    CollectiveAtoms &atoms,
    std::map<AxisKey, TypedValue<axis::AxisTypeInterface>> &axisProvenance,
    size_t &nextReplicateId, CollectiveResolutionError &error) {
  // Resolves a group's factors onto axes; extent-1 factors carry no data and
  // are dropped. `tileSpace` is the tile a shape-axis factor refers to.
  auto resolveGroup = [&](TV_FactorGroup group,
                          AtomSpace tileSpace) -> FailureOr<ResolvedGroup> {
    auto factors = axis::getProductProvenanceFactors(group);
    if (failed(factors)) {
      error.reasons.push_back("factor group must be produced by axis.product");
      return failure();
    }
    ResolvedGroup resolved;
    for (TV_AxisFactor factor : *factors) {
      uint64_t extent = axis::getFactorExtent(factor);
      if (extent == 1)
        continue;
      auto provenance = axis::getFactorProvenanceAxis(factor);
      if (failed(provenance)) {
        error.reasons.push_back("factor has no provenance axis");
        return failure();
      }
      AxisKey key;
      Type type = provenance->getType();
      if (auto physical = dyn_cast<PhysicalCommAxisType>(type)) {
        auto meshIdx = llvm::find(meshAxisTypes, physical);
        if (meshIdx == meshAxisTypes.end()) {
          error.reasons.push_back("factor over an axis outside the module's "
                                  "mesh");
          return failure();
        }
        key = {AtomSpace::Mesh,
               static_cast<size_t>(meshIdx - meshAxisTypes.begin())};
      } else if (isa<axis::ShapeAxisType>(type)) {
        key = {tileSpace,
               static_cast<size_t>(axis::getAxisDimIndex(
                   cast<TypedValue<axis::ShapeAxisType>>(*provenance)))};
      } else if (isa<ReplicationAxisType>(type)) {
        key = {AtomSpace::Replicate, nextReplicateId++};
        atoms.addAxis(key, axis::getAxisExtent(*provenance));
      } else {
        error.reasons.push_back("expected a fully-lowered physical, tensor, "
                                "or replication axis");
        return failure();
      }
      ResolvedFactor resolvedFactor{
          key, extent, static_cast<uint64_t>(axis::getFactorStride(factor)),
          *provenance};
      atoms.addFactor(resolvedFactor);
      resolved.push_back(resolvedFactor);
      // try_emplace keeps the first provenance seen for a key, matching this
      // function's resolution order (reduction groups, then each mapping
      // pair's lhs and rhs, then the mesh operands) -- every factor over the
      // same key shares one provenance axis, so which occurrence wins
      // doesn't otherwise matter.
      axisProvenance.try_emplace(key, *provenance);
    }
    return resolved;
  };

  CollectiveFactors result;
  for (Value group : collective.getReductionGroups()) {
    auto resolved = resolveGroup(cast<TV_FactorGroup>(group), spaces.input);
    if (failed(resolved))
      return failure();
    result.reductionGroups.push_back(std::move(*resolved));
  }

  auto mapOp = collective.getMapping().getDefiningOp<axis::AxisMapOp>();
  if (!mapOp) {
    error.reasons.push_back("collective mapping must be produced by axis.map");
    return failure();
  }
  for (auto [lhsGroup, rhsGroup] : mapOp.getTypedMappingPairs()) {
    // Both sides are resolved before either result is checked, so a pair with
    // problems on both sides reports both.
    auto lhs = resolveGroup(lhsGroup, spaces.input);
    auto rhs = resolveGroup(rhsGroup, spaces.output);
    if (failed(lhs) || failed(rhs))
      return failure();
    result.pairs.push_back({std::move(*lhs), std::move(*rhs)});
  }

  // The mesh operands are resolved last (see resolveCollectiveAtoms's header
  // comment for why the replicate-id order doesn't matter in practice): their
  // factors are registered as cut sources exactly like any other, but kept
  // out of reductionGroups/pairs since they mean neither a reduction nor a
  // relabeling. The tile space is never used for these, since a mesh
  // operand's factors are always physical-axis provenance (see
  // MakeReplicationsExplicit.cpp, the sole builder of these operands).
  auto inputMeshFactors =
      resolveGroup(collective.getInputMesh(), AtomSpace::Mesh);
  if (failed(inputMeshFactors))
    return failure();
  result.inputMeshFactors = std::move(*inputMeshFactors);
  auto outputMeshFactors =
      resolveGroup(collective.getOutputMesh(), AtomSpace::Mesh);
  if (failed(outputMeshFactors))
    return failure();
  result.outputMeshFactors = std::move(*outputMeshFactors);
  return result;
}

FailureOr<CollectiveResolution> resolveCollectiveAtoms(
    DistributedCollectiveOp collective,
    ArrayRef<PhysicalCommAxisType> meshAxisTypes, ArrayRef<int64_t> inputTile,
    ArrayRef<int64_t> outputTile, CollectiveResolutionError &error) {
  error = CollectiveResolutionError();

  CollectiveResolution resolution;
  CollectiveAtoms &atoms = resolution.atoms;
  for (size_t a = 0; a < meshAxisTypes.size(); ++a)
    atoms.addAxis({AtomSpace::Mesh, a}, meshAxisTypes[a].getExtent());
  // InTile and OutTile are independent axis spaces (one indexed by the input
  // tensor's own rank, the other by the output's), so they are registered in
  // separate loops rather than one shared by index -- a collective may
  // reshape and have the two tiles differ in rank.
  for (size_t j = 0; j < inputTile.size(); ++j)
    atoms.addAxis({AtomSpace::InTile, j}, inputTile[j]);
  for (size_t j = 0; j < outputTile.size(); ++j)
    atoms.addAxis({AtomSpace::OutTile, j}, outputTile[j]);

  size_t nextReplicateId = 0;
  auto factors = resolveCollectiveFactors(
      collective, meshAxisTypes, {AtomSpace::InTile, AtomSpace::OutTile}, atoms,
      resolution.axisProvenance, nextReplicateId, error);
  if (failed(factors))
    return failure();
  resolution.reductionGroups = std::move(factors->reductionGroups);
  resolution.pairs = std::move(factors->pairs);
  resolution.inputMeshFactors = std::move(factors->inputMeshFactors);
  resolution.outputMeshFactors = std::move(factors->outputMeshFactors);

  if (failed(atoms.refine(resolution.pairs))) {
    error.kind = CollectiveResolutionError::Kind::NoCommonAtoms;
    error.reasons.push_back("a mapping pair is indivisible, or an axis's "
                            "factor boundaries are not nested");
    return failure();
  }
  return resolution;
}

namespace {

// Builds a fresh axis.factor over one atom, at the builder's current
// position: the digit the atom names is, by construction, exactly the
// extent/stride absolute cut `resolution.atoms` computed for it, over the
// axis its originating factor came from. Callers skip extent-1 atoms, which
// carry no data and would otherwise need a provenance value even for tile
// axes no factor in the collective ever references.
//
// A replicate atom is handled differently from a mesh/tile one: a replicate
// axis has no identity worth preserving across factors (resolution treats
// every replicate factor as its own one-off axis sized to that factor's own
// extent), so a strided sub-factor of the original, possibly coarser
// replicate axis would misrepresent this atom's size the moment anything
// re-resolves it. A fresh, exactly-sized ReplicationAxis stands in for it,
// the same "just need something with the same extent" idiom
// distributed-make-replications-explicit uses.
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

// One fresh factor per atom in `labels` whose extent is greater than one, in
// order.
SmallVector<TV_AxisFactor>
buildAtomFactors(const CollectiveResolution &resolution,
                 ArrayRef<AtomLabel> labels, OpBuilder &builder, Location loc) {
  SmallVector<TV_AxisFactor> factors;
  for (const AtomLabel &label : labels)
    if (resolution.atoms.extentOf(label) != 1)
      factors.push_back(buildAtomFactor(resolution, label, builder, loc));
  return factors;
}

} // namespace

AtomicOperands buildAtomicOperands(const CollectiveResolution &resolution,
                                   ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                   OpBuilder &builder, Location loc) {
  const CollectiveAtoms &atoms = resolution.atoms;
  AtomicOperands operands;

  // Every op built below is pure axis-algebra metadata and belongs at module
  // scope.
  axis::ModuleScopeGuard moduleScope(builder);

  // input_mesh/output_mesh: every mesh atom, axis 0's atoms then axis 1's,
  // and so on, assembled into one product. Built twice (once per side)
  // rather than once and reused, since the two operands are logically
  // independent even when they end up structurally identical (a later cse
  // deduplicates them).
  auto buildMeshOperand = [&] {
    SmallVector<TV_AxisFactor> factors;
    for (size_t a = 0; a < meshAxisTypes.size(); ++a)
      factors.append(buildAtomFactors(
          resolution, atoms.labelsOfAxis({AtomSpace::Mesh, a}), builder, loc));
    return Value(axis::viewFactorsAsProduct(factors, builder, loc));
  };
  operands.inputMesh = buildMeshOperand();
  operands.outputMesh = buildMeshOperand();

  // Each reduction group's factors, atomized in place and reassembled into
  // one product per group.
  for (const ResolvedGroup &group : resolution.reductionGroups) {
    SmallVector<TV_AxisFactor> factors;
    for (const ResolvedFactor &factor : group)
      factors.append(
          buildAtomFactors(resolution, atoms.labelsOf(factor), builder, loc));
    operands.reductionGroups.push_back(
        axis::viewFactorsAsProduct(factors, builder, loc));
  }

  // Mapping: one atomic (single-factor) pair per atom position of every
  // original pair, all collected into ONE fresh axis.map, never one axis.map
  // per original pair. A pair's lhs/rhs atom runs are positionally aligned
  // one-to-one by computeCommonAtomsAndMappingSplits, so the two extents at a
  // given position always agree and either both or neither side is skipped.
  SmallVector<Value> mappingLhs, mappingRhs;
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
      mappingLhs.push_back(axis::viewFactorsAsProduct(lhsFactor, builder, loc));
      mappingRhs.push_back(axis::viewFactorsAsProduct(rhsFactor, builder, loc));
    }
  }
  operands.mapping = builder
                         .create<axis::AxisMapOp>(loc, ValueRange(mappingLhs),
                                                  ValueRange(mappingRhs))
                         .getMap();
  return operands;
}

bool isCollectiveAtomic(const CollectiveResolution &resolution) {
  const CollectiveAtoms &atoms = resolution.atoms;
  auto oneAtom = [&](const ResolvedFactor &factor) {
    return atoms.labelsOf(factor).size() == 1;
  };
  auto allOneAtom = [&](const ResolvedGroup &group) {
    return llvm::all_of(group, oneAtom);
  };

  for (const ResolvedGroup &group : resolution.reductionGroups)
    if (!allOneAtom(group))
      return false;
  // A mapping pair also needs exactly one factor per side: a two-factor
  // side where each factor happens to already be one atom still isn't one
  // factor to one factor.
  for (const auto &[lhs, rhs] : resolution.pairs)
    if (lhs.size() != 1 || rhs.size() != 1 || !oneAtom(lhs.front()) ||
        !oneAtom(rhs.front()))
      return false;
  if (!allOneAtom(resolution.inputMeshFactors) ||
      !allOneAtom(resolution.outputMeshFactors))
    return false;

  // Every mesh axis this collective's own mesh operands ever registered a
  // factor for must have every one of its atoms covered, exactly once, by
  // inputMeshFactors, and likewise by outputMeshFactors. Since every factor
  // checked above is already known to be one atom, "exactly once" reduces to
  // set equality between the atoms a side's factors reference and the axis's
  // full atom set.
  std::set<size_t> meshAxes;
  for (const ResolvedFactor &factor : resolution.inputMeshFactors)
    meshAxes.insert(factor.key.second);
  for (const ResolvedFactor &factor : resolution.outputMeshFactors)
    meshAxes.insert(factor.key.second);

  auto coversAxisExactly = [&](const ResolvedGroup &meshFactors, size_t axis) {
    std::set<size_t> covered;
    size_t factorCount = 0;
    for (const ResolvedFactor &factor : meshFactors) {
      if (factor.key.second != axis)
        continue;
      ++factorCount;
      covered.insert(atoms.labelsOf(factor).front().atom);
    }
    // A duplicate reference to one atom would collapse into `covered`
    // without changing its size, so comparing sizes catches "not once" the
    // same way set equality below catches "not every atom".
    if (covered.size() != factorCount)
      return false;
    std::set<size_t> expected;
    for (const AtomLabel &label : atoms.labelsOfAxis({AtomSpace::Mesh, axis}))
      expected.insert(label.atom);
    return covered == expected;
  };
  for (size_t axis : meshAxes)
    if (!coversAxisExactly(resolution.inputMeshFactors, axis) ||
        !coversAxisExactly(resolution.outputMeshFactors, axis))
      return false;

  return true;
}

} // namespace mlir::enzyme::distributed
