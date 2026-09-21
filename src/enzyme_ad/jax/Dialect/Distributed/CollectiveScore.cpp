#include "CollectiveScore.h"

#include "Utilities.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::distributed {

namespace {

void appendLabel(llvm::raw_ostream &os, const AtomLabel &label) {
  os << static_cast<int>(label.space) << "." << label.axis << "." << label.atom;
}

void appendRole(llvm::raw_ostream &os, const MeshAtomRole &role) {
  os << static_cast<int>(role.kind) << "(";
  appendLabel(os, role.partner);
  os << ")";
}

// A string that is equal for two (collective, params) pairs exactly when
// every field the planner reads is equal.
std::string cacheKey(const NormalizedCollective &collective,
                     const MeshCostParams &params) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << "p";
  for (size_t i = 0; i < params.numAxes(); ++i)
    os << " " << params.axisExtents[i] << "/" << params.bandwidth[i] << "/"
       << params.roundLatency[i];
  os << " L" << params.launchLatency << " R"
     << static_cast<int>(collective.reductionKind) << " B"
     << collective.payloadBytes;
  for (const MeshAtom &atom : collective.meshAtoms) {
    os << " m" << atom.axis << "." << atom.atom << "x" << atom.extent << "s"
       << atom.stride << ":";
    appendRole(os, atom.in);
    os << ">";
    appendRole(os, atom.out);
  }
  for (const TileAtom &tile : collective.tileAtoms) {
    os << " t" << tile.isInput << tile.dim << "." << tile.atom << "x"
       << tile.extent;
    if (tile.partner) {
      os << "~";
      appendLabel(os, *tile.partner);
    }
  }
  return key;
}

} // namespace

CollectiveCostSummary CollectiveCostModel::summarize(ModuleOp module) {
  CollectiveCostSummary summary;

  FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
  if (failed(physicalMesh)) {
    summary.failureReason = "no unique physical mesh";
    return summary;
  }
  SmallVector<PhysicalCommAxisType> meshAxisTypes;
  std::vector<uint64_t> extents;
  for (Attribute axisAttr : physicalMesh->getAxesAttr()) {
    auto axis = cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue());
    meshAxisTypes.push_back(axis);
    extents.push_back(axis.getExtent());
  }
  // Uniform bandwidth and latencies on every axis.
  MeshCostParams params(extents);

  module.walk([&](DistributedCollectiveOp collective) {
    ++summary.numCollectives;
    if (!summary.feasible())
      return;
    std::string reason;
    std::optional<NormalizedCollective> normalized =
        normalizeCollective(collective, meshAxisTypes, reason);
    if (!normalized) {
      summary.failureReason = reason.empty() ? "normalization failed" : reason;
      return;
    }

    auto [it, inserted] = cache.try_emplace(cacheKey(*normalized, params));
    Entry &entry = it->second;
    if (inserted) {
      ++misses;
      std::optional<std::vector<PrimitiveStep>> chain =
          plan(*normalized, params, entry.failureReason);
      if (chain) {
        entry.ok = true;
        for (const PrimitiveStep &step : *chain)
          entry.duration += step.isolatedDuration();
      } else if (entry.failureReason.empty()) {
        entry.failureReason = "planning failed";
      }
    } else {
      ++hits;
    }

    if (!entry.ok) {
      summary.failureReason = entry.failureReason;
      return;
    }
    summary.durations.push_back(entry.duration);
    summary.total += entry.duration;
  });
  return summary;
}

} // namespace mlir::enzyme::distributed
