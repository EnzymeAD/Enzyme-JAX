#include "CollectiveCost.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>

namespace mlir::enzyme::distributed {

MeshCostParams::MeshCostParams(std::vector<uint64_t> extents)
    : axisExtents(std::move(extents)),
      bandwidth(axisExtents.size(), kDefaultBandwidth),
      roundLatency(axisExtents.size(), kDefaultRoundLatency),
      launchLatency(kDefaultLaunchLatency) {}

MeshCostParams::MeshCostParams(std::vector<uint64_t> extents,
                               std::vector<double> bw,
                               std::vector<double> latency, double launch)
    : axisExtents(std::move(extents)), bandwidth(std::move(bw)),
      roundLatency(std::move(latency)), launchLatency(launch) {
  assert(bandwidth.size() == axisExtents.size() &&
         roundLatency.size() == axisExtents.size());
}

const char *toString(PrimitiveKind kind) {
  switch (kind) {
  case PrimitiveKind::AllReduce:
    return "all-reduce";
  case PrimitiveKind::ReduceScatter:
    return "reduce-scatter";
  case PrimitiveKind::AllGather:
    return "all-gather";
  case PrimitiveKind::AllToAll:
    return "all-to-all";
  case PrimitiveKind::Permute:
    return "permute";
  case PrimitiveKind::LocalSlice:
    return "local-slice";
  }
  llvm_unreachable("unknown PrimitiveKind");
}

namespace {

uint64_t groupSize(const std::vector<StepAtom> &atoms) {
  uint64_t n = 1;
  for (const StepAtom &atom : atoms)
    n *= atom.extent;
  return n;
}

// Fills the derived fields (N1: BW[a] is the port bandwidth; N4: axes are
// independent, so the slowest axis sets the transfer time) (transferTime, rho)
// from `step.volume`.
void finishStep(PrimitiveStep &step, const MeshCostParams &params) {
  step.rho.assign(params.numAxes(), 0.0);
  step.transferTime = 0;
  for (size_t a = 0; a < params.numAxes(); ++a)
    step.transferTime =
        std::max(step.transferTime, step.volume[a] / params.bandwidth[a]);
  if (step.transferTime > 0)
    for (size_t a = 0; a < params.numAxes(); ++a)
      step.rho[a] = (step.volume[a] / params.bandwidth[a]) / step.transferTime;
}

uint64_t ceilLog2(uint64_t n) {
  uint64_t k = 0;
  while ((uint64_t{1} << k) < n)
    ++k;
  return k;
}

// A step over one physical axis moving `volume` bytes per port (N3) in
// `rounds` rounds (N6). A step that moves nothing (rounds == 0) has no
// latency.
PrimitiveStep singleAxisStep(PrimitiveKind kind,
                             const std::vector<StepAtom> &atoms,
                             int64_t payloadIn, int64_t payloadOut,
                             double volume, uint64_t rounds,
                             const MeshCostParams &params) {
  assert(!atoms.empty() && "step without atoms");
  size_t axis = atoms.front().axis;
  assert(axis < params.numAxes() && "atom outside the mesh");
  for (const StepAtom &atom : atoms)
    assert(atom.axis == axis && "step atoms must share one physical axis");
  PrimitiveStep step;
  step.kind = kind;
  step.atoms = atoms;
  step.payloadIn = payloadIn;
  step.payloadOut = payloadOut;
  step.volume.assign(params.numAxes(), 0.0);
  step.volume[axis] = volume;
  step.latency =
      rounds == 0 ? 0.0
                  : params.launchLatency + params.roundLatency[axis] * rounds;
  finishStep(step, params);
  return step;
}

// Ties go to `bandwidthOptimal`.
PrimitiveStep cheaper(PrimitiveStep bandwidthOptimal, PrimitiveStep other) {
  return other.isolatedDuration() < bandwidthOptimal.isolatedDuration()
             ? other
             : bandwidthOptimal;
}

} // namespace

PrimitiveStep allReduceFootprint(const std::vector<StepAtom> &atoms,
                                 int64_t payload,
                                 const MeshCostParams &params) {
  uint64_t n = groupSize(atoms);
  uint64_t k = ceilLog2(n);
  PrimitiveStep halvingDoubling =
      singleAxisStep(PrimitiveKind::AllReduce, atoms, payload, payload,
                     2.0 * payload * (n - 1) / n, 2 * k, params);
  PrimitiveStep recursiveDoubling =
      singleAxisStep(PrimitiveKind::AllReduce, atoms, payload, payload,
                     1.0 * payload * k, k, params);
  return cheaper(halvingDoubling, recursiveDoubling);
}

PrimitiveStep reduceScatterFootprint(const std::vector<StepAtom> &atoms,
                                     int64_t payload,
                                     const MeshCostParams &params) {
  uint64_t n = groupSize(atoms);
  assert(payload % n == 0 && "reduce-scatter payload must divide by n");
  return singleAxisStep(PrimitiveKind::ReduceScatter, atoms, payload,
                        payload / n, 1.0 * payload * (n - 1) / n, ceilLog2(n),
                        params);
}

PrimitiveStep allGatherFootprint(const std::vector<StepAtom> &atoms,
                                 int64_t payload,
                                 const MeshCostParams &params) {
  uint64_t n = groupSize(atoms);
  return singleAxisStep(PrimitiveKind::AllGather, atoms, payload, payload * n,
                        1.0 * payload * (n - 1), ceilLog2(n), params);
}

PrimitiveStep allToAllFootprint(const std::vector<StepAtom> &atoms,
                                int64_t payload, const MeshCostParams &params) {
  uint64_t n = groupSize(atoms);
  uint64_t k = ceilLog2(n);
  PrimitiveStep pairwise =
      singleAxisStep(PrimitiveKind::AllToAll, atoms, payload, payload,
                     1.0 * payload * (n - 1) / n, n - 1, params);
  PrimitiveStep bruck = singleAxisStep(PrimitiveKind::AllToAll, atoms, payload,
                                       payload, 0.5 * payload * k, k, params);
  return cheaper(pairwise, bruck);
}

PrimitiveStep permuteFootprint(const std::vector<StepAtom> &atoms,
                               const std::vector<double> &changeFraction,
                               int64_t payload, const MeshCostParams &params) {
  assert(changeFraction.size() == atoms.size());
  PrimitiveStep step;
  step.kind = PrimitiveKind::Permute;
  step.atoms = atoms;
  step.payloadIn = payload;
  step.payloadOut = payload;
  step.volume.assign(params.numAxes(), 0.0);
  // N5: the round latency of an axis is paid once however many atoms of the
  // step lie on it, and only by axes a message actually crosses.
  std::vector<bool> crossed(params.numAxes(), false);
  for (size_t i = 0; i < atoms.size(); ++i) {
    assert(atoms[i].axis < params.numAxes() && "atom outside the mesh");
    double v = payload * changeFraction[i];
    step.volume[atoms[i].axis] = std::max(step.volume[atoms[i].axis], v);
    crossed[atoms[i].axis] = crossed[atoms[i].axis] || v > 0;
  }
  double roundLatencySum = 0;
  bool moves = false;
  for (size_t a = 0; a < params.numAxes(); ++a)
    if (crossed[a]) {
      moves = true;
      roundLatencySum += params.roundLatency[a];
    }
  step.latency = moves ? params.launchLatency + roundLatencySum : 0.0;
  finishStep(step, params);
  return step;
}

double permuteSwapChangeFraction(uint64_t e) { return 1.0 - 1.0 / e; }

PrimitiveStep localSliceFootprint(const std::vector<StepAtom> &atoms,
                                  int64_t payloadIn, int64_t payloadOut,
                                  const MeshCostParams &params) {
  PrimitiveStep step;
  step.kind = PrimitiveKind::LocalSlice;
  step.atoms = atoms;
  step.payloadIn = payloadIn;
  step.payloadOut = payloadOut;
  step.volume.assign(params.numAxes(), 0.0);
  finishStep(step, params);
  return step;
}

} // namespace mlir::enzyme::distributed
