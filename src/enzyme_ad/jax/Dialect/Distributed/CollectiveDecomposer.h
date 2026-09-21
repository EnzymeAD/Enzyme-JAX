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
//   (Reduced, Replicate)              all-reduce
//   (Reduced, Tile)                   reduce-scatter, or all-reduce + local
//                                     slice
//   (Replicate, Tile)                 local slice
//   (Tile, Replicate)                 all-gather
//   (Tile, Tile)                      all-to-all, or all-gather + local slice
//   (Mesh, Mesh) in pure cycles       collective permute
//   (Replicate, Replicate), or a mesh atom feeding itself: nothing
// Every other row with a Mesh role, (Reduced, Mesh), (Replicate, Mesh),
// (Tile, Mesh), (Mesh, Tile), (Mesh, Replicate) and a (Mesh, Mesh) atom whose
// component is not a pure cycle, is mesh-coupled: its cost depends on the
// partner atoms' roles. Mesh-coupled components are decomposed by the
// half-split (D11), which is the baseline every collective supports, and are
// also offered cheaper variants: fused exchanges (D14), a path closure (D15)
// and a conjugation onto a faster axis (D16). Tile atoms paired only with
// tile atoms of the other side are device-local relabelings and cost
// nothing. Not decomposed, and reported by plan() as a failure with a
// reason:
//   - reductions whose body is not a single recognized associative operation
//     (D9).
//   - collectives with more than kMaxLiveUnits units (D7).
//   - tile atoms whose partner is neither a mesh atom nor the other side's
//     tile.
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
//       by e, a gather multiplies it by e, an all-to-all, an all-reduce and a
//       permute keep it. The payload before a step therefore depends only on
//       which steps were done, not on their order (N7: size is the only cost
//       input).
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
//       searched. Reduce-scatter atoms have the same three stages as
//       tile-to-tile atoms (see D10).
//   D8  Every reduced atom is its own unit and its own single-axis step,
//       (Reduced, Replicate) an all-reduce and (Reduced, Tile) a reduce-
//       scatter onto that atom's tile atom (N1, N5). Reduction steps are not
//       merged across atoms or split into rounds, so several reduced atoms are
//       ordered by the DP like any other units (D6). A reduce-scatter divides
//       the payload by its extent like a slice (D3) but costs time; an
//       all-reduce keeps the payload. Combining values is free (N8).
//   D9  Only reductions whose body is a single recognized associative and
//       commutative kind (add, min, max, mul, and, or, xor) are decomposed,
//       because the log-round algorithms combine partial results in an
//       arbitrary order (N2). Other bodies make plan() fail.
//   D10 A (Reduced, Tile) atom also offers an all-reduce followed by the free
//       local slice (D2) as a DP branch, in the way gather + slice is one for
//       tile-to-tile atoms (D4). It is offered so the checker and the
//       print pass can exercise it, but it is never cheaper: an
//       all-reduce moves about twice the volume of a reduce-scatter (N9's
//       formulas) in no fewer rounds, so a cost-minimizing search never
//       selects it.
//   D11 Mesh atoms tied together by mesh partners form a connected component.
//       A component of only (Mesh, Mesh) atoms is a set of permutation cycles
//       (D5). Any other component is half-split: an atom with roles (in = P,
//       out = Q) becomes a first half by P (all-reduce for Reduced, all-gather
//       for Tile or Mesh, nothing for Replicate) that leaves the atom holding
//       replicated data, then a free local slice (D2) placing Q (nothing for
//       Replicate). This can move more data than a dedicated move of the
//       digit would, but it uses only the primitives above.
//   D12 A unit may depend on other units. A half-split slice waits for its own
//       atom's first half and, when its output digit comes from a partner atom,
//       for the partner's first half, so that the digit is local by then.
//       Dependencies restrict the DP's orders (D6) but do not change costs.
//       A gather can therefore raise the payload above its initial value
//       before the slice that consumes it lowers it again (D3).
//   D13 Components share only the payload (and, see D16, the atoms they
//       borrow). Each lists alternative variants (sets of units), and plan()
//       searches every combination of one variant per component and keeps the
//       cheapest chain. The first variant is the half-split, which is what
//       keeps every collective supported; when the product of the variant
//       counts would exceed kMaxVariantCombinations, later variants of later
//       components are dropped, so the bound holds in release builds too.
//   D14 Fused exchange. In a half-split, an atom that gathers its own digit
//       (in is Tile or Mesh) and then slices a digit onto itself (out is Tile
//       or Mesh) is a TileToTile unit: besides gather + slice it may run one
//       all-to-all on its axis, which keeps the payload where the gather
//       would multiply it by the extent (D4 applies unchanged). A reduced atom
//       placing a digit (out is Mesh) is likewise a ReduceScatter unit. The
//       digit an exchange or slice places on the atom must already be local,
//       so it waits for the unit of the atom that owns that digit; this
//       needs only that unit to have communicated (D12), by gather or by
//       exchange. An exchange placing a digit that came from another atom
//       therefore follows that atom's gather. The atom's own gather does not
//       wait.
//   D15 Path closure. A component that is not a pure cycle is one path along
//       its mesh links: data flows from a start atom s (out is Tile or
//       Replicate) through atoms that hand their digit on to an end atom e
//       (in is Reduced, Tile or Replicate). When e holds nothing, every atom
//       takes its source's digit and s takes e's empty content, which is one
//       cyclic shift of the atoms' contents: a single permute over the whole
//       path with change fraction permuteSwapChangeFraction(extent) on each
//       atom (D5, N5), not the S (n - 1) of a gather. Before it, e is emptied
//       (all-reduce for Reduced, gather for Tile, nothing for Replicate); after
//       it, a free slice places s's tile digit. The permute is a variant of
//       the component beside the half-split, so it is used only when cheaper.
//       A network where a message may not cross several axes at once
//       (replacing N5) would change the permute's cost, not its validity.
//   D16 Conjugation. A gather or all-reduce on atom A of a slow axis may run
//       on an atom M instead when M has the same extent, holds no digit and
//       has no role (in and out are Replicate), and its axis has strictly
//       more bandwidth (a different bandwidth class, N1). A swap permute of A
//       and M moves A's digit or pending reduction onto M, the step runs on M,
//       and since M then holds nothing and A holds M's empty content, both
//       atoms are in the state the unconjugated step would have left and no
//       swap back is needed. The step costs what it costs on M's axis; the
//       swap crosses both axes (N5) and moves the payload at that moment.
//       Each variant conjugates one unit, the fastest such M is chosen, and
//       two components never borrow the same M. Steps that leave a digit on the
//       atom (all-to-all, reduce-scatter, slices) are not conjugated: after
//       the swap they would place the digit of the atom whose content moved,
//       which the chain checker cannot tell from the step alone.

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
  // Offer the variants that beat the half-split for mesh-coupled components
  // (D14 to D16). Off, mesh-coupled components are only half-split with plain
  // gathers and slices, which lets tests pin the baseline.
  bool relayVariants = true;
};

// Most combinations of component variants (see D13) plan() searches.
constexpr size_t kMaxVariantCombinations = 64;

// The most units of work (see D7) plan() accepts. The memoized state space
// grows up to 3^units, so a collective near this limit can take very long to
// search exactly. TODO: replace with a state budget once the heuristic step
// orderer bounds the branching.
constexpr size_t kMaxLiveUnits = 20;

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
// an all-to-all does both on one atom, and a permute moves the contents of
// atoms: either the collective's own permutation cycles and paths (a path
// closes into a cycle through its empty end atom), or a swap of two atoms that
// runs a step on another atom instead (D16). The digit a slice or all-to-all
// places on atom `a` is the one the collective's output role for `a`
// requires: an input tile digit, or, for a Mesh(m) role, the digit that
// started on input atom m, which must already be local (a gather of m moved
// it there). A gather of an atom whose input role is Mesh releases that
// atom's own digit the same way as for a Tile role.
//
// A reduced input atom carries no digit; its digit is summed away by exactly
// one all-reduce (the atom then holds replicated data, and a slice may follow)
// or reduce-scatter (the atom then holds the tile digit its output role
// requires). A reduction step on an atom that is not reduced, or that was
// already reduced, fails, as does a chain that leaves a reduced atom
// unreduced or a collective whose reduction body is not a recognized kind.
//
// The chain passes if every step's preconditions hold, each step's payloads
// equal the independently recomputed tile size, and the final placement equals
// the collective's required placement (which also drops digits replicated away
// and requires the right output tile). Returns false and sets `why` otherwise.
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
