#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_ATOMS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_ATOMS_H

#include "Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include <map>
#include <optional>
#include <string>
#include <utility>

namespace mlir::enzyme::distributed {

// Index spaces a collective's factors can slice. Mesh axes are shared by the
// input and output; tensor dimensions differ between the input tile and the
// output tile, so each side gets its own space. Every replicate factor is an
// independent one-off axis of its own. MidTile is the tile between two chained
// collectives (the first's output, the second's input); it only exists when a
// chained pair is resolved jointly (see resolveCollectiveFactors), so that the
// fused collective's own InTile/OutTile spaces stay meaningful.
enum class AtomSpace { Mesh, InTile, OutTile, Replicate, MidTile };
using AxisKey = std::pair<AtomSpace, size_t>;

// One atom (indivisible digit) of an axis, identified by its position among
// that axis's atoms, major-first.
struct AtomLabel {
  AtomSpace space;
  size_t axis;
  size_t atom;

  bool operator==(const AtomLabel &other) const {
    return space == other.space && axis == other.axis && atom == other.atom;
  }
};

// A factor of a collective's reduction group, mapping, or mesh operand,
// resolved to the axis it slices.
struct ResolvedFactor {
  AxisKey key;
  uint64_t extent;
  uint64_t stride;
  // The canonical axis this factor is a piece of (axis::
  // getFactorProvenanceAxis's result for it). Not used by any resolution
  // logic here; carried along so a caller rebuilding axis algebra from a
  // resolved factor (the atomizing rewrite) can create a fresh axis.factor
  // over the same source axis without re-deriving provenance.
  TypedValue<axis::AxisTypeInterface> provenanceAxis;
};
using ResolvedGroup = SmallVector<ResolvedFactor>;

// The common atoms of every factor a collective mentions (see
// axis::computeCommonAtomsAndMappingSplits), addressed by axis keys.
//
// Reshaping an expanded tensor to one dim per atom exposes every factor as
// whole dims, and a mapping pair then becomes a per-atom relabeling: the k-th
// atom of its lhs pairs with the k-th of its rhs.
//
// Assumes each group's index space is row-major over its factors,
// major-first.
class CollectiveAtoms {
public:
  void addAxis(AxisKey key, uint64_t extent) {
    auto [it, inserted] = index.try_emplace(key, keys.size());
    if (inserted) {
      keys.push_back(key);
      extents.push_back(extent);
    } else {
      extents[it->second] = extent;
    }
  }

  void addFactor(const ResolvedFactor &factor) {
    others.push_back(toAtomFactor(factor));
  }

  LogicalResult
  refine(ArrayRef<std::pair<ResolvedGroup, ResolvedGroup>> pairs) {
    SmallVector<axis::AtomFactorPair> atomPairs;
    for (const auto &[lhs, rhs] : pairs)
      atomPairs.push_back({toAtomGroup(lhs), toAtomGroup(rhs)});
    auto result =
        axis::computeCommonAtomsAndMappingSplits(extents, atomPairs, others);
    if (failed(result))
      return failure();
    atoms.emplace(std::move(*result));
    return success();
  }

  SmallVector<AtomLabel> labelsOf(const ResolvedFactor &factor) const {
    auto [first, count] = atoms->rangeOf(toAtomFactor(factor));
    SmallVector<AtomLabel> labels;
    for (size_t i = first; i < first + count; ++i)
      labels.push_back({factor.key.first, factor.key.second, i});
    return labels;
  }

  SmallVector<AtomLabel> labelsOf(const ResolvedGroup &group) const {
    SmallVector<AtomLabel> labels;
    for (const ResolvedFactor &factor : group)
      labels.append(labelsOf(factor));
    return labels;
  }

  SmallVector<AtomLabel> labelsOfAxis(AxisKey key) const {
    SmallVector<AtomLabel> labels;
    for (size_t i = 0; i < atoms->atomsOf(index.at(key)).size(); ++i)
      labels.push_back({key.first, key.second, i});
    return labels;
  }

  uint64_t extentOf(const AtomLabel &label) const {
    return atoms->atomsOf(index.at({label.space, label.axis}))[label.atom]
        .extent;
  }

  uint64_t strideOf(const AtomLabel &label) const {
    return atoms->atomsOf(index.at({label.space, label.axis}))[label.atom]
        .stride;
  }

  SmallVector<int64_t> extentsOf(ArrayRef<AtomLabel> labels) const {
    SmallVector<int64_t> result;
    for (const AtomLabel &label : labels)
      result.push_back(extentOf(label));
    return result;
  }

private:
  std::map<AxisKey, size_t> index;
  SmallVector<AxisKey> keys;
  SmallVector<uint64_t> extents;
  SmallVector<axis::AtomFactor> others;
  std::optional<axis::CommonAtoms> atoms;

  axis::AtomFactor toAtomFactor(const ResolvedFactor &factor) const {
    return {index.at(factor.key), factor.extent, factor.stride};
  }
  axis::AtomFactorGroup toAtomGroup(const ResolvedGroup &group) const {
    axis::AtomFactorGroup result;
    for (const ResolvedFactor &factor : group)
      result.push_back(toAtomFactor(factor));
    return result;
  }
};

// A collective's factors resolved onto keyed axes and refined to common atoms.
struct CollectiveResolution {
  // One group per reduction group, in operand order.
  SmallVector<ResolvedGroup> reductionGroups;
  // One (lhs, rhs) group pair per mapping pair, in mapping order.
  SmallVector<std::pair<ResolvedGroup, ResolvedGroup>> pairs;
  // The collective's own input_mesh/output_mesh factors, resolved the same
  // way as any other factor and folded into `atoms` as cut sources (so the
  // common atom basis also respects a mesh operand's own cuts), but kept
  // separate from reductionGroups/pairs since they are neither a reduction
  // nor a relabeling: existing callers that only care about those two
  // meanings are unaffected by this field's contents.
  SmallVector<ResolvedFactor> inputMeshFactors;
  SmallVector<ResolvedFactor> outputMeshFactors;
  // The provenance value recorded for each axis key, from whichever resolved
  // factor over it was encountered first. Every mesh axis and every tile
  // axis with extent greater than 1 is guaranteed an entry (see
  // resolveCollectiveAtoms's own resolution order below): mesh axes through
  // inputMeshFactors/outputMeshFactors, tile axes through reductionGroups or
  // pairs. Consumed by the atomizing rewrite in AtomizeCollectives.cpp to
  // build a fresh axis.factor over an atom without re-deriving provenance.
  std::map<AxisKey, TypedValue<axis::AxisTypeInterface>> axisProvenance;
  CollectiveAtoms atoms;
};

// One collective's factors resolved onto keyed axes, before any refinement.
struct CollectiveFactors {
  SmallVector<ResolvedGroup> reductionGroups;
  SmallVector<std::pair<ResolvedGroup, ResolvedGroup>> pairs;
  ResolvedGroup inputMeshFactors;
  ResolvedGroup outputMeshFactors;
};

// Which axis space a collective's tile factors belong to: `input` for
// reduction groups and mapping lhs groups, `output` for mapping rhs groups.
struct TileSpaces {
  AtomSpace input;
  AtomSpace output;
};

// Why resolveCollectiveAtoms failed. The kind tells the caller how to treat it.
struct CollectiveResolutionError {
  enum class Kind {
    // The collective is malformed for this resolution, or the module is not
    // fully lowered: a group or mapping not built from axis.product / axis.map,
    // a factor with no provenance axis, a physical axis outside the module's
    // mesh, or an axis that is not physical, tensor or replication (logical or
    // device-local residue). `reasons` has one message per problem found.
    Structural,
    // Every factor resolved but no common atom basis exists: a mapping pair is
    // indivisible or an axis's cuts do not nest, as for a 3*2 factor pair
    // mapped against 2*3. Valid input can end up here. `reasons` has one
    // message that completes "could not split into common atoms (...)".
    NoCommonAtoms,
  };
  Kind kind = Kind::Structural;
  SmallVector<std::string> reasons;
};

// Resolves every factor `collective` mentions onto keyed axes and computes
// their common atoms.
//
// The axes are the module's mesh axes (shared by input and output), each
// tile dimension of the input and of the output, and one axis per replicate
// factor. Each factor of a reduction group, a mapping pair, or the
// collective's own input_mesh/output_mesh operand is resolved to one of these
// and registered as a cut source, so the common atom basis respects a mesh
// operand's own cuts too; extent-1 factors carry no data and are dropped.
// Replicate axes are numbered in resolution order: reduction groups first,
// then each mapping pair's lhs followed by its rhs, then input_mesh and
// output_mesh last. (In practice a mesh operand holds only physical factors
// once distributed-make-replications-explicit has run, so this last step
// rarely allocates a replicate id at all; the order only matters for
// determinism.)
//
// `meshAxisTypes` is the module's physical mesh; `inputTile` and `outputTile`
// are the local tile shapes on either side. They index independent axis
// spaces (InTile/OutTile) and need not have equal rank -- a collective may
// reshape, as when a gather's output has more dims than its per-device input.
//
// Assumes each group's index space is row-major over its factors,
// major-first.
//
// Emits no diagnostics. On failure `error` says which kind of failure
// occurred and why, so the caller picks its own severity.
FailureOr<CollectiveResolution> resolveCollectiveAtoms(
    DistributedCollectiveOp collective,
    ArrayRef<PhysicalCommAxisType> meshAxisTypes, ArrayRef<int64_t> inputTile,
    ArrayRef<int64_t> outputTile, CollectiveResolutionError &error);

// The building block of resolveCollectiveAtoms, exposed so that several
// collectives can be resolved into one shared atom space (the fusion of a
// chained pair does this, giving the first's output tile and the second's
// input tile the same axes).
//
// Resolves every factor of `collective` (reduction groups first, then each
// mapping pair's lhs and rhs, then input_mesh and output_mesh, the order that
// numbers replicate axes) and registers each as a cut source of `atoms`, which
// must already have its mesh and tile axes added. `axisProvenance` and
// `nextReplicateId` are shared across calls so replicate axes of different
// collectives stay distinct. Does not refine `atoms`; the caller passes every
// pair of every collective sharing the space to CollectiveAtoms::refine.
// `error` is appended to, not reset.
FailureOr<CollectiveFactors> resolveCollectiveFactors(
    DistributedCollectiveOp collective,
    ArrayRef<PhysicalCommAxisType> meshAxisTypes, TileSpaces spaces,
    CollectiveAtoms &atoms,
    std::map<AxisKey, TypedValue<axis::AxisTypeInterface>> &axisProvenance,
    size_t &nextReplicateId, CollectiveResolutionError &error);

// The axis-algebra operands of an atomic collective.
struct AtomicOperands {
  Value inputMesh;
  Value outputMesh;
  Value mapping;
  // One product per entry of resolution.reductionGroups.
  SmallVector<Value> reductionGroups;
};

// Builds fresh axis.factor/axis.product/axis.map values expressing
// `resolution` at atom granularity: one factor per atom (extent-1 atoms
// dropped), input_mesh/output_mesh covering every atom of every mesh axis,
// one product per reduction group, and one single-factor pair per atom
// position of every mapping pair, all in one axis.map. The values are built
// at module scope and are never shared with existing IR, so a caller can
// assign them to a collective without disturbing others that use the old
// operands. Replicate atoms get a fresh, exactly-sized ReplicationAxis.
//
// `resolution` needs `atoms`, `axisProvenance`, `reductionGroups` and `pairs`
// (each group/pair's atoms must have equal extents position by position);
// `builder`'s location only matters for op locations.
AtomicOperands buildAtomicOperands(const CollectiveResolution &resolution,
                                   ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                   OpBuilder &builder, Location loc);

// Whether every factor `resolution` resolved already IS one atom -- the
// property distributed-atomize-collectives's rewrite establishes, and
// nothing here re-derives: a factor resolves to more than one atom exactly
// when some other factor mentioned anywhere in the collective cuts partway
// through it, which is what property (iii) ("equal or disjoint") rules out.
// Given that, a mesh operand whose factors are each already one atom is
// automatically disjoint (distinct atoms never overlap) and, since a mesh
// axis is always registered at its full physical extent (regardless of what
// the mesh operand's own factors happen to cover -- see
// resolveCollectiveAtoms's own initial atoms.addAxis calls), covering every
// one of its atoms is exactly covering the whole axis: this is property (i).
bool isCollectiveAtomic(const CollectiveResolution &resolution);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_ATOMS_H
