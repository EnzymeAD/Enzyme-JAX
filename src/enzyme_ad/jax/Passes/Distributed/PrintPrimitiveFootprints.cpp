#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveCost.h"

#include <algorithm>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_PRINTPRIMITIVEFOOTPRINTSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct PrintPrimitiveFootprintsPass
    : public impl::PrintPrimitiveFootprintsPassBase<
          PrintPrimitiveFootprintsPass> {
  using PrintPrimitiveFootprintsPassBase::PrintPrimitiveFootprintsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (axes.size() != extents.size()) {
      module.emitError("axes and extents must have the same length");
      return signalPassFailure();
    }

    if (axes.empty()) {
      module.emitError("at least one atom is required");
      return signalPassFailure();
    }
    for (size_t i = 0; i < axes.size(); ++i)
      if (axes[i] < 0 || extents[i] < 1) {
        module.emitError("axes must be non-negative and extents positive");
        return signalPassFailure();
      }

    std::vector<StepAtom> atoms;
    size_t numAxes = bandwidths.size();
    for (size_t i = 0; i < axes.size(); ++i) {
      atoms.push_back(
          {static_cast<size_t>(axes[i]), i, static_cast<uint64_t>(extents[i])});
      if (bandwidths.empty())
        numAxes = std::max(numAxes, static_cast<size_t>(axes[i]) + 1);
    }
    if (!atoms.empty() && kind != "collective-permute" && kind != "local-slice")
      for (const StepAtom &atom : atoms)
        if (atom.axis != atoms.front().axis) {
          module.emitError("'") << kind << "' atoms must share one axis";
          return signalPassFailure();
        }
    for (const StepAtom &atom : atoms)
      if (atom.axis >= numAxes) {
        module.emitError("atom axis outside the mesh");
        return signalPassFailure();
      }

    // Extents of the whole axes are not needed by the closed forms.
    MeshCostParams params(
        std::vector<uint64_t>(numAxes, 1),
        bandwidths.empty()
            ? std::vector<double>(numAxes, 1.0)
            : std::vector<double>(bandwidths.begin(), bandwidths.end()),
        std::vector<double>(numAxes, latency),
        MeshCostParams::kDefaultLaunchLatency);

    PrimitiveStep step;
    if (kind == "all-reduce") {
      step = allReduceFootprint(atoms, payloadBytes, params);
    } else if (kind == "reduce-scatter") {
      step = reduceScatterFootprint(atoms, payloadBytes, params);
    } else if (kind == "all-gather") {
      step = allGatherFootprint(atoms, payloadBytes, params);
    } else if (kind == "all-to-all") {
      step = allToAllFootprint(atoms, payloadBytes, params);
    } else if (kind == "collective-permute") {
      if (changedFractions.size() != atoms.size()) {
        module.emitError(
            "collective-permute needs one changed fraction per atom");
        return signalPassFailure();
      }
      step = permuteFootprint(
          atoms,
          std::vector<double>(changedFractions.begin(), changedFractions.end()),
          payloadBytes, params);
    } else if (kind == "local-slice") {
      step = localSliceFootprint(atoms, payloadBytes, payloadBytes, params);
    } else {
      module.emitError("unknown primitive kind '") << kind << "'";
      return signalPassFailure();
    }
    module.emitRemark() << describeStep(step);
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
