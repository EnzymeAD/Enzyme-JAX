#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/ComputeCost.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_PRINTCOMPUTECOSTPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Compact decimal, so hand-computed values can be matched textually.
std::string shortest(double value) {
  std::string text;
  llvm::raw_string_ostream(text) << llvm::format("%g", value);
  return text;
}

struct PrintComputeCostPass
    : public impl::PrintComputeCostPassBase<PrintComputeCostPass> {
  using PrintComputeCostPassBase::PrintComputeCostPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // A debug pass on hand-written inputs, so the mesh is optional here.
    DeviceCostParams device;
    auto meshes = module.getOps<PhysicalMeshOp>();
    if (!meshes.empty())
      device = deviceCostParamsFromPhysicalMesh(*meshes.begin());
    if (flops >= 0)
      device.flops = flops;
    if (memBandwidth >= 0)
      device.memBandwidth = memBandwidth;

    ComputeCostSummary summary = summarizeKernelCompute(
        module, device, [&](Operation *op, const OpCost &cost) {
          op->emitRemark() << "flops=" << shortest(cost.flops)
                           << " bytes=" << shortest(cost.bytes)
                           << " time=" << shortest(cost.time(device))
                           << (cost.unknown ? " (unknown op)" : "");
        });
    module.emitRemark() << "compute cost: total=" << shortest(summary.total)
                        << " over " << summary.numKernels << " kernels, "
                        << summary.numOps
                        << " ops (unknown: " << summary.numUnknownOps
                        << ", collective: " << summary.numCollectiveOps
                        << ", skipped: " << summary.numSkippedOps << ")";
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
