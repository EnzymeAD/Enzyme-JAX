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
// independent one-off axis of its own.
enum class AtomSpace { Mesh, InTile, OutTile, Replicate };
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

// A factor of a collective's reduction group or mapping, resolved to the
// axis it slices.
struct ResolvedFactor {
  AxisKey key;
  uint64_t extent;
  uint64_t stride;
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
  CollectiveAtoms atoms;
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
// factor. Each factor of a reduction group or mapping is resolved to one of
// these and registered as a cut source, and extent-1 factors carry no data and
// are dropped. Replicate axes are numbered in resolution order: reduction
// groups first, then each mapping pair's lhs followed by its rhs.
//
// `meshAxisTypes` is the module's physical mesh; `inputTile` and `outputTile`
// are the local tile shapes on either side and must have equal rank.
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

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_ATOMS_H
