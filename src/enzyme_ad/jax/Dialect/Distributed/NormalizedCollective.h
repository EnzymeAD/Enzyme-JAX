#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_NORMALIZED_COLLECTIVE_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_NORMALIZED_COLLECTIVE_H

#include "CollectiveAtoms.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {

// The normal form of an atomic distributed.Collective, as consumed by the
// collective decomposer and cost model.
//
// NormalizedCollective and everything it contains is a plain value type with
// no MLIR values, operations or types, so it can be copied, hashed and
// memoized independently of the IR it came from.
//
// A mapping pair (L -> R) sends input digit L to output digit R. Every mesh
// atom (indivisible digit of a physical mesh axis) has an `in` role (where its
// input digit goes) and an `out` role (where its output digit comes from).
// Together they say which communication primitive the atom needs (see RolePair
// and classify).

// Where a mesh atom's digit goes (`in`) or comes from (`out`).
//   in  Tile: the pair is (mesh atom -> OutTile atom): data along the mesh atom
//       is gathered into the local tile.
//   out Tile: the pair is (InTile atom -> mesh atom): the tile atom is
//       sliced or scattered onto the mesh atom.
enum class AtomRole {
  // In only: the atom is a member of a reduction group.
  Reduced,
  // The digit is the tile atom `partner` (in: an OutTile atom the digit goes
  // to, out: an InTile atom the digit comes from).
  Tile,
  // The digit goes to / comes from the other mesh atom `partner`.
  Mesh,
  // The digit is broadcast to, or drawn from, a replication.
  Replicate,
};

struct MeshAtomRole {
  AtomRole kind;
  // Meaningful for Tile and Mesh only.
  AtomLabel partner{AtomSpace::Mesh, 0, 0};
};

// A mesh atom, with its extent and stride within its physical axis. Extent-1
// atoms are omitted, as they carry no data.
struct MeshAtom {
  size_t axis;
  size_t atom;
  uint64_t extent;
  uint64_t stride;
  MeshAtomRole in;
  MeshAtomRole out;
};

// An atom of the per-device input or output tile (extent-1 atoms omitted).
// `partner` is the atom on the other end of the pair that mentions it, if any:
// a mesh atom (the tile digit becomes / comes from a mesh digit), a tile atom
// of the other side (a device-local relabeling that involves no mesh
// communication), or a replicate atom.
struct TileAtom {
  bool isInput;
  size_t dim;
  size_t atom;
  uint64_t extent;
  std::optional<AtomLabel> partner;
};

// How a collective's reduction body combines values. None means the
// collective reduces nothing; Unknown is a body that is not a single
// recognized associative op.
enum class ReductionKind { None, Unknown, Add, Min, Max, Mul, And, Or, Xor };

struct NormalizedCollective {
  // Every atom of every mesh axis the collective's mesh operands touch,
  // ordered by axis, then atom (major-first).
  std::vector<MeshAtom> meshAtoms;
  // Input-tile atoms in dim order, then output-tile atoms in dim order. Tile
  // atoms paired only with other tile atoms are kept here rather than in a
  // separate list: they are free, but their extents decide which tile atoms a
  // peeling reduce-scatter may target.
  std::vector<TileAtom> tileAtoms;
  // Bytes of the per-device input tile, the payload every primitive of the
  // decomposition starts from.
  int64_t payloadBytes = 0;
  // The one kind shared by all reduction groups, or None without a reduction.
  // Collectives whose groups use different bodies are unsupported.
  ReductionKind reductionKind = ReductionKind::None;
};

// The communication primitive a mesh atom's (in, out) roles call for.
enum class RolePair {
  AllReduce,         // (Reduced, Replicate)
  ReduceScatter,     // (Reduced, Tile)
  ReduceThenPermute, // (Reduced, Mesh)
  AllGather,         // (Tile, Replicate)
  TileToTile,        // (Tile, Tile): all-to-all, or gather then slice
  LocalSlice,        // (Replicate, Tile): free
  NoOp,              // (Replicate, Replicate) or a mesh atom feeding itself
  Permute,           // every other combination involving a mesh partner
};

// Maps an atom's roles to the primitive they call for.
//
// Combinations with no dedicated row that involve a mesh partner are all
// Permute: (Tile, Mesh), (Replicate, Mesh), (Mesh, Tile), (Mesh, Replicate)
// and (Mesh, Mesh) between two different atoms. Some of these also imply a
// slice, gather or broadcast, and their cost depends on the partner atom's
// role, so the decomposer resolves them per connected component (a pure
// permutation cycle stays a permute, anything else is half-split). `atom`'s own
// identity is needed to tell a mesh atom feeding itself (NoOp) from a genuine
// permutation.
RolePair classify(const MeshAtom &atom);

const char *toString(AtomRole role);
const char *toString(RolePair pair);
const char *toString(ReductionKind kind);

// Builds the normal form of an atomic collective.
//
// `resolution` must satisfy isCollectiveAtomic. Every mesh atom must appear
// on both sides of a mapping pair or reduction (explicit replication has run),
// and reductions must be over mesh atoms only; violations are asserted, not
// reported. `reductionKind` must be None exactly when there are no reduction
// groups.
NormalizedCollective
buildNormalizedCollective(const CollectiveResolution &resolution,
                          ArrayRef<int64_t> inputTile,
                          ArrayRef<int64_t> outputTile, int64_t elementBytes,
                          ReductionKind reductionKind);

// Reads the tile shapes, element width and reduction kinds off `collective`,
// resolves it, and builds its normal form. On failure (a structural
// resolution error, no common atoms, or a resolution that is not atomic)
// returns nullopt and sets `failureReason`.
std::optional<NormalizedCollective>
normalizeCollective(DistributedCollectiveOp collective,
                    ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                    std::string &failureReason);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_NORMALIZED_COLLECTIVE_H
