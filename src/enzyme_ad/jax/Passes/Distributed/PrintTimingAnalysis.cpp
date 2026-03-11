#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_PRINTTIMINGANALYSISPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct PrintTimingAnalysisPass
    : public enzyme::distributed::impl::PrintTimingAnalysisPassBase<
          PrintTimingAnalysisPass> {
  using PrintTimingAnalysisPassBase::PrintTimingAnalysisPassBase;

  void runOnOperation() override {
    RegionComputationOp regionOp = getOperation();

    const auto &hb = getAnalysis<HappensBeforeAnalysis>();
    const auto &timing = getAnalysis<TimingAnalysis>();

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
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir