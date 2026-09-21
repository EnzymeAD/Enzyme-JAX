#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_COST_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_COST_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir::enzyme::distributed {

// Closed-form cost footprints of the primitive collectives a decomposed
// distributed.Collective is made of.
//
// Everything here is a plain value type with no MLIR values, operations or
// types. PrimitiveStep in particular is consumed by cost models outside this
// file, so its field names are stable.
//
// Network model
// -------------
// Every footprint formula rests on the assumptions below. Each formula's
// comment names the labels it depends on and what would change if one of them
// were replaced, so the model can be altered assumption by assumption.
//   N1  Each physical mesh axis is fully connected: any two devices on it
//       communicate at equal cost through a port of bandwidth BW[a] per
//       device. There is no ring or other topology.
//   N2  Log-round algorithms (recursive halving and doubling, Bruck) are
//       available on every axis and take ceil(log2 n) rounds, also for group
//       sizes that are not powers of two.
//   N3  V[a] is the bytes one device sends (and receives) through its port on
//       axis a. Steps do not contend or share ports with each other; sharing
//       is a matter for whatever composes steps.
//   N4  A device's traffic on different axes is independent, with no shared
//       injection limit.
//   N5  Multi-axis routing is dimension ordered: a message that changes
//       several axes' digits crosses each of those axes, using that axis's
//       port and paying that axis's round latency. Only a permute spans axes.
//   N6  One launch latency per collective and one round latency per axis.
//       They default to uniform constants until mesh metadata provides them.
//   N7  Payload bytes are the only size input: no message-size-dependent
//       bandwidth, protocol switch or minimum message size.
//   N8  Combining values in a reduction is free; compute is not modelled.
//   N9  Where an algorithm is chosen inside a footprint (all-reduce,
//       all-to-all), the criterion is the isolated duration
//       L + max_a V[a] / BW[a], ties going to the bandwidth-optimal form.
//
// Quantities
// ----------
//   V[a]   per-port bytes on axis a during the step (N3).
//   L      latency: launch latency plus round latency times rounds (N6).
//   The isolated duration of a step is L + max_a V[a] / BW[a]. Its relative
//   demand rho[a] = (V[a] / BW[a]) / max_b (V[b] / BW[b]) says what fraction
//   of the step's slowest-axis time each axis is busy, which is what a
//   contention model needs to overlap other work into the unused headroom.
//
// All-reduce, reduce-scatter, all-gather and all-to-all steps run over one
// physical axis. A collective spanning several axes is a chain of such steps.

// Bandwidth and latency of a physical mesh, indexed by physical axis.
//
// The vectors are public so a caller can build parameters from any source.
// A later constructor will read them from the physical mesh's bandwidth and
// latency metadata; until that metadata exists, the extents-only constructor
// gives uniform parameters that make hand computation easy.
//
// Latency model (N6): a collective takes some number of rounds (see the
// footprint functions), each costing the per-round latency of its axis, on top
// of one launch latency. A group of size 1 launches nothing and has zero
// latency.
struct MeshCostParams {
  // Extent of each physical mesh axis (not used by the closed forms; kept so
  // consumers can size arrays and validate atoms against the mesh).
  std::vector<uint64_t> axisExtents;
  // Bytes per unit time of one device's port on each axis (N1).
  std::vector<double> bandwidth;
  // Latency of one communication round on each axis.
  std::vector<double> roundLatency;
  // Fixed cost of launching any collective that moves data.
  double launchLatency;

  static constexpr double kDefaultBandwidth = 1.0;
  static constexpr double kDefaultRoundLatency = 0.01;
  static constexpr double kDefaultLaunchLatency = 0.1;

  // Uniform parameters: every axis has bandwidth 1 and round latency 0.01,
  // and a launch costs 0.1.
  explicit MeshCostParams(std::vector<uint64_t> axisExtents);

  // Explicit per-axis parameters; the three vectors have one entry per axis.
  MeshCostParams(std::vector<uint64_t> axisExtents,
                 std::vector<double> bandwidth,
                 std::vector<double> roundLatency, double launchLatency);

  size_t numAxes() const { return axisExtents.size(); }
};

enum class PrimitiveKind {
  AllReduce,
  ReduceScatter,
  AllGather,
  AllToAll,
  Permute,
  // A device-local slice of a replicated dimension. It communicates nothing,
  // so chain assembly can treat every stage of a decomposition uniformly.
  LocalSlice,
};

const char *toString(PrimitiveKind kind);

// A mesh atom a step communicates over. `atom` indexes the axis's atoms as in
// AtomLabel, so consumers can relate a step to a NormalizedCollective.
struct StepAtom {
  size_t axis;
  size_t atom;
  uint64_t extent;
};

// One primitive collective with its footprint.
struct PrimitiveStep {
  PrimitiveKind kind;
  // The atoms the step communicates over. Atoms sharing a physical axis
  // count once toward that axis's volume.
  std::vector<StepAtom> atoms;
  // Per-device bytes before and after the step.
  int64_t payloadIn = 0;
  int64_t payloadOut = 0;
  // Launch plus per-round latency.
  double latency = 0;
  // Per-port bytes on each physical axis (N3), indexed by axis; zero for axes
  // the step does not touch.
  std::vector<double> volume;
  // Relative demand per axis, in [0, 1]; all zero when the step moves no data.
  std::vector<double> rho;
  // max_a V[a] / BW[a]: the time the slowest axis is busy.
  double transferTime = 0;

  // L + max_a V[a] / BW[a].
  double isolatedDuration() const { return latency + transferTime; }
};

// Footprints. `atoms` lists the mesh atoms of the communicating group, and the
// group size n is the product of their extents. For every kind but permute all
// atoms must lie on one physical axis (asserted). Atom strides do not matter
// on a fully connected axis, so the atoms need not be contiguous. `payload` is
// the per-device input bytes of the step, S. For n = 1 every footprint has
// zero volume and latency. With k = ceil(log2 n):

// All-reduce: the cheaper of two algorithms (N9, ties to the first).
// Depends on N1 (any pair can exchange directly, so halving and doubling need
// no multi-hop), N2 (round counts), N3, N6, N7 and N8.
//   halving + doubling (reduce-scatter then all-gather): 2k rounds,
//     V = 2 S (n - 1) / n
//   recursive doubling (exchange the whole payload each round): k rounds,
//     V = S k
// Output payload S. On a ring (replacing N1) the volume would stay
// 2 S (n - 1) / n but the rounds would grow to 2 (n - 1) and recursive
// doubling would no longer exist.
PrimitiveStep allReduceFootprint(const std::vector<StepAtom> &atoms,
                                 int64_t payload, const MeshCostParams &params);

// Reduce-scatter (recursive halving): k rounds, V = S (n - 1) / n, S the input
// payload. Output payload S / n (S must be divisible by n). Depends on N1, N2,
// N3, N6, N7, N8. On a ring the rounds would be n - 1, with V unchanged.
PrimitiveStep reduceScatterFootprint(const std::vector<StepAtom> &atoms,
                                     int64_t payload,
                                     const MeshCostParams &params);

// All-gather (recursive doubling): k rounds. Each device receives (n - 1)
// shards, so V = payloadIn (n - 1) = payloadOut (n - 1) / n. Output payload
// n * payloadIn. The (n - 1) / n factor applies to the larger output side;
// `payload` here is the small input shard. Depends on N1, N2, N3, N6, N7. On a
// ring the rounds would be n - 1, with V unchanged.
PrimitiveStep allGatherFootprint(const std::vector<StepAtom> &atoms,
                                 int64_t payload, const MeshCostParams &params);

// All-to-all: the cheaper of two algorithms (N9, ties to the first). Depends
// on N1 (pairwise exchange needs a direct path to every peer), N2 (Bruck
// rounds), N3, N6, N7.
//   pairwise exchange: n - 1 rounds, V = S (n - 1) / n
//   Bruck: k rounds, V = (S / 2) k
// Output payload S. S is the input payload. Bruck trades volume for rounds; a
// model with size-dependent protocols (replacing N7) could shift the
// crossover.
PrimitiveStep allToAllFootprint(const std::vector<StepAtom> &atoms,
                                int64_t payload, const MeshCostParams &params);

// Permute: one round in which a fraction of devices send their whole payload.
// Depends on N1, N3, N4, N5, N6. A message may cross several axes, so the
// round costs the launch latency plus the SUM of the round latencies of the
// axes with a nonzero change fraction (N5); with a non-dimension-ordered
// route it would be a max instead.
// V[a] = S * changeFraction[a] where changeFraction[a] is the fraction of
// devices whose digit on the step's atom over axis a changes. The caller
// supplies it per atom (parallel to `atoms`); for two atoms of equal extent e
// swapping digits it is permuteSwapChangeFraction(e) on both. If two atoms
// share an axis, the larger fraction is used.
PrimitiveStep permuteFootprint(const std::vector<StepAtom> &atoms,
                               const std::vector<double> &changeFraction,
                               int64_t payload, const MeshCostParams &params);

// Fraction of devices whose digit changes when two digits of extent `e` swap
// values: those whose two digits differ, 1 - 1 / e.
double permuteSwapChangeFraction(uint64_t e);

// Device-local slice: zero cost, payload shrinks from `payloadIn` to
// `payloadOut`.
PrimitiveStep localSliceFootprint(const std::vector<StepAtom> &atoms,
                                  int64_t payloadIn, int64_t payloadOut,
                                  const MeshCostParams &params);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_COST_H
