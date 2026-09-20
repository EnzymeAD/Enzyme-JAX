#include "CollectiveAtoms.h"

namespace mlir::enzyme::distributed {

namespace {
using TV_AxisFactor = TypedValue<axis::AxisFactorType>;
using TV_FactorGroup = TypedValue<axis::FactorGroupType>;
} // namespace

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

  // Resolves a group's factors onto axes; extent-1 factors carry no data and
  // are dropped. `tileSpace` is the tile a shape-axis factor refers to.
  size_t nextReplicateId = 0;
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
    }
    return resolved;
  };

  for (Value group : collective.getReductionGroups()) {
    auto resolved =
        resolveGroup(cast<TV_FactorGroup>(group), AtomSpace::InTile);
    if (failed(resolved))
      return failure();
    resolution.reductionGroups.push_back(std::move(*resolved));
  }

  auto mapOp = collective.getMapping().getDefiningOp<axis::AxisMapOp>();
  if (!mapOp) {
    error.reasons.push_back("collective mapping must be produced by axis.map");
    return failure();
  }
  for (auto [lhsGroup, rhsGroup] : mapOp.getTypedMappingPairs()) {
    // Both sides are resolved before either result is checked, so a pair with
    // problems on both sides reports both.
    auto lhs = resolveGroup(lhsGroup, AtomSpace::InTile);
    auto rhs = resolveGroup(rhsGroup, AtomSpace::OutTile);
    if (failed(lhs) || failed(rhs))
      return failure();
    resolution.pairs.push_back({std::move(*lhs), std::move(*rhs)});
  }

  // The mesh operands are resolved last (see this function's header comment
  // for why the replicate-id order doesn't matter in practice): their
  // factors are registered as cut sources exactly like any other, but kept
  // out of reductionGroups/pairs since they mean neither a reduction nor a
  // relabeling. `tileSpace` is passed but never actually used for these,
  // since a mesh operand's factors are always physical-axis provenance (see
  // MakeReplicationsExplicit.cpp, the sole builder of these operands).
  auto inputMeshFactors =
      resolveGroup(collective.getInputMesh(), AtomSpace::Mesh);
  if (failed(inputMeshFactors))
    return failure();
  resolution.inputMeshFactors = std::move(*inputMeshFactors);
  auto outputMeshFactors =
      resolveGroup(collective.getOutputMesh(), AtomSpace::Mesh);
  if (failed(outputMeshFactors))
    return failure();
  resolution.outputMeshFactors = std::move(*outputMeshFactors);

  if (failed(atoms.refine(resolution.pairs))) {
    error.kind = CollectiveResolutionError::Kind::NoCommonAtoms;
    error.reasons.push_back("a mapping pair is indivisible, or an axis's "
                            "factor boundaries are not nested");
    return failure();
  }
  return resolution;
}

} // namespace mlir::enzyme::distributed
