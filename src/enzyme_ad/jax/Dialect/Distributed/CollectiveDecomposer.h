#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_DECOMPOSER_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_DECOMPOSER_H

#include "CollectiveCost.h"
#include "NormalizedCollective.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {

// Decomposes an atomic distributed.Collective into a chain of primitive
// collectives (PrimitiveStep, see CollectiveCost.h) chosen by isolated
// duration.
//
// plan() works on a NormalizedCollective and MeshCostParams only, so it has no
// MLIR values in it. planCollective() is the thin IR-facing wrapper that also
// reports where the chain attaches to the surrounding program.
//
// Supported collectives
// ---------------------
// Rows of the normal form (see RolePair) per mesh atom:
//   (Replicate, Tile)                 local slice
//   (Tile, Replicate)                 all-gather
//   (Tile, Tile)                      all-to-all, or all-gather + local slice
//   (Mesh, Mesh), different atoms     collective permute
//   (Replicate, Replicate), or a mesh atom feeding itself: nothing
// Tile atoms paired only with tile atoms of the other side are device-local
// relabelings and cost nothing. Reductions and the mixed rows (Tile, Mesh),
// (Replicate, Mesh), (Mesh, Tile), (Mesh, Replicate) are not decomposed yet;
// plan() reports them as a failure with a reason rather than approximating.
//
// Decomposition assumptions
// -------------------------
// They extend the network model N1..N9 of CollectiveCost.h. Each names the
// labels it depends on.
//   D1  One step per mesh atom for gathers, all-to-alls and slices, running
//       on that atom's axis (N1, N5: only a permute spans axes). Atoms of
//       the same axis are not merged into one step and steps are not split
//       into rounds. Merging cannot lower latency on a fully connected axis
//       (N2) and only saves one launch (N6), so it would be a refinement of
//       the same decomposition.
//   D2  Local slices communicate nothing, only shrink the payload, so they
//       are applied as soon as they are available and never branch. Tile
//       atoms paired with tile atoms of the other side are relabelings of
//       local layout, which is not costed.
//   D3  Payload tracking: a slice on extent e divides the per-device payload
//       by e, a gather multiplies it by e, an all-to-all and a permute keep
//       it. The payload before a step therefore depends only on which steps
//       were done, not on their order (N7: size is the only cost input).
//   D4  Tile-to-tile atoms choose between one all-to-all and a gather
//       followed by the (free) slice. Both are DP branches, not a choice
//       made inside a footprint, because their payload effects on the other
//       atoms' steps differ (N9 applies only within one footprint).
//   D5  All (Mesh, Mesh) atoms of nontrivial cycles form one permute step.
//       Its change fraction per atom is permuteSwapChangeFraction(extent):
//       an atom's digit is replaced by another atom's digit of equal extent,
//       so it changes for all devices whose two digits differ (N5).
//   D6  The objective is the sum of the steps' isolated durations. Steps of
//       one chain are treated as strictly sequential and not overlapped
//       with each other (N3: no sharing between steps). Because payload
//       depends only on the done set (D3), the minimum over orders is an
//       exact dynamic program over the per-atom stage.
//   D7  The DP state is one stage per unit of work (a slice, gather,
//       tile-to-tile atom or the permute group). Tile-to-tile atoms have an
//       intermediate stage, gathered with the slice pending, and the pending
//       slice is a free step available immediately (D2). A collective with
//       more than kMaxLiveUnits units is reported as unsupported rather than
//       searched.

// Per-unit progress of the decomposition. Two states with equal stages are
// equal for the purposes of every later decision (D3), so it is the memo key.
struct DecomposerState {
  std::vector<uint8_t> stage;
  bool operator<(const DecomposerState &other) const {
    return stage < other.stage;
  }
  bool operator==(const DecomposerState &other) const {
    return stage == other.stage;
  }
};

// One available next step and the state it leads to.
struct CandidateStep {
  PrimitiveStep step;
  DecomposerState next;
};

// Hook applied to the candidate list of every state that has more than one
// candidate, before the DP expands it. It may reorder or truncate the list but
// must keep it non-empty. A state whose only choice is a free step (D2) is
// not offered to the hook. The default (no hook) expands every candidate, which
// makes plan() the exact minimum.
using CandidateFilter =
    std::function<void(const DecomposerState &, std::vector<CandidateStep> &)>;

struct PlanOptions {
  CandidateFilter filter;
};

// The most units of work (see D7) plan() accepts. The memoized state space
// grows up to 3^units, so this keeps the exact DP tractable.
constexpr size_t kMaxLiveUnits = 12;

// The chain realizing `collective` with the least summed isolated duration
// (D6), in execution order. An empty chain means the collective moves nothing
// and is the identity. Returns nullopt and sets `failureReason` for
// collectives outside the supported rows. Debug builds assert that the chain
// passes verifyChainRealizesCollective.
std::optional<std::vector<PrimitiveStep>>
plan(const NormalizedCollective &collective, const MeshCostParams &params,
     std::string &failureReason, const PlanOptions &options = {});

// Symbolically executes `chain` and checks that it realizes `collective`,
// independently of how plan() searched.
//
// The model tracks where each data digit lives. Digits are the input mesh
// atoms whose data varies along them (roles Tile and Mesh) and the input tile
// atoms. A digit is either held on a mesh atom (its coordinate selects which
// piece a device has) or local to the tile. Gathers move a digit from a mesh
// atom into the tile, slices move a required tile digit onto a free mesh atom,
// an all-to-all does both on one atom, and a permute renames mesh atoms. The
// digit a slice or all-to-all places on atom `a` is the one the collective's
// output role for `a` requires. The chain passes if every step's preconditions
// hold, each step's payloads equal the independently recomputed tile size,
// and the final placement equals the collective's required placement (which
// also drops digits replicated away and requires the right output tile).
// Returns false and sets `why` otherwise.
bool verifyChainRealizesCollective(const NormalizedCollective &collective,
                                   const std::vector<PrimitiveStep> &chain,
                                   std::string &why);

// The decomposition of one distributed.Collective together with the SSA values
// it attaches to. The chain is strictly sequential, so these two ports and the
// step order are the only dependencies a whole-program scheduler needs:
//
//   inputPort   feeds the first step (the collective's input_object operand;
//               it is defined outside the chain).
//   resultPort  is produced by the last step (the Await's result; every use
//               of it depends on the last step finishing).
//
// An empty chain connects inputPort straight to resultPort. The handle
// between `collective` and `await` is internal to the chain and has no
// consumers other than `await`.
struct CollectivePlan {
  DistributedCollectiveOp collective;
  DistributedAwait await;
  Value inputPort;
  Value resultPort;
  std::vector<PrimitiveStep> chain;

  double totalDuration() const;
};

// Normalizes `collective` and decomposes it. Returns nullopt and sets
// `failureReason` if it cannot be normalized (see normalizeCollective) or is
// outside the supported rows (see plan).
std::optional<CollectivePlan>
planCollective(DistributedCollectiveOp collective,
               ArrayRef<PhysicalCommAxisType> meshAxisTypes,
               const MeshCostParams &params, std::string &failureReason,
               const PlanOptions &options = {});

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_DECOMPOSER_H
