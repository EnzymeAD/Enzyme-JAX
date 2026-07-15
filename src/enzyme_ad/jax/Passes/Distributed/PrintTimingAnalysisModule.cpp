#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_PRINTTIMINGANALYSISMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

void printTimingAnalysisForMesh(MeshComputationOp meshOp) {
  HappensBeforeAnalysis hb(meshOp);
  UnitTimingCostModel costModel;
  TimingAnalysis timing(hb, costModel);

  llvm::outs() << "Timing analysis for " << meshOp.getOperationName() << "\n";

  unsigned laneIndex = 0;
  for (Region *lane : meshOp.getLanes()) {
    llvm::outs() << "  lane " << laneIndex++ << ":\n";
    for (Operation &op : lane->getOps()) {
      TimingAnalysis::TimeRange timeRange = timing.getTimeRange(&op);
      llvm::outs() << "    [" << timeRange.first << ", " << timeRange.second
                   << ") ";
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

    SmallVector<MeshComputationOp> meshOps;
    moduleOp.walk([&](MeshComputationOp meshOp) { meshOps.push_back(meshOp); });

    for (MeshComputationOp meshOp : meshOps)
      printTimingAnalysisForMesh(meshOp);
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir