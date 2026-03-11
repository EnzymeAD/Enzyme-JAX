#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_PRINTTIMINGANALYSISMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

void printTimingAnalysisForRegion(RegionComputationOp regionOp) {
  HappensBeforeAnalysis hb(regionOp);
  UnitTimingCostModel costModel;
  TimingAnalysis timing(hb, costModel);

  llvm::outs() << "Timing analysis for " << regionOp.getOperationName()
               << "\n";

  unsigned laneIndex = 0;
  for (Region *lane : regionOp.getLanes()) {
    llvm::outs() << "  lane " << laneIndex++ << ":\n";
    for (Operation &op : lane->getOps()) {
      TimingAnalysis::TimeRange timeRange = timing.getTimeRange(&op);
      llvm::outs() << "    [" << timeRange.first << ", "
                   << timeRange.second << ") ";
      if (Operation *root = hb.classRoot(&op)) {
        if (root != &op)
          llvm::outs() << "(rooted at " << root->getName() << ") ";
      }
      op.print(llvm::outs(), OpPrintingFlags().skipRegions());
      llvm::outs() << "\n";
    }
  }
}

struct PrintTimingAnalysisModulePass
    : public enzyme::distributed::impl::PrintTimingAnalysisModulePassBase<
          PrintTimingAnalysisModulePass> {
  using PrintTimingAnalysisModulePassBase::PrintTimingAnalysisModulePassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    SmallVector<RegionComputationOp> regionOps;
    moduleOp.walk([&](RegionComputationOp regionOp) { regionOps.push_back(regionOp); });

    for (RegionComputationOp regionOp : regionOps)
      printTimingAnalysisForRegion(regionOp);
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir