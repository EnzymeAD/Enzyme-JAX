#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"

#include <algorithm>

namespace mlir {
namespace enzyme {
namespace distributed {

int64_t UnitTimingCostModel::getOperationDuration(Operation *op) const {
  (void)op;
  return 1;
}

TimingAnalysis::TimingAnalysis(Operation *op, AnalysisManager &am) {
  (void)op;
  const auto &hbAnalysis = am.getAnalysis<HappensBeforeAnalysis>();
  hb = &hbAnalysis;
  UnitTimingCostModel unitCostModel;
  buildTimingMap(hbAnalysis, unitCostModel);
}

TimingAnalysis::TimingAnalysis(const HappensBeforeAnalysis &hb,
                               const TimingCostModel &costModel) {
  this->hb = &hb;
  buildTimingMap(hb, costModel);
}

void TimingAnalysis::buildTimingMap(const HappensBeforeAnalysis &hb,
                                    const TimingCostModel &costModel) {
  rootToTimeRange.clear();

  // classesInTopologicalOrder() guarantees predecessors come before successors,
  // so a single forward pass is sufficient to compute start/end times.
  for (Operation *root : hb.classesInTopologicalOrder()) {
    int64_t duration = 0;
    for (Operation *member : hb.classList(root)) {
      duration = std::max(duration, costModel.getOperationDuration(member));
    }

    int64_t startTime = 0;
    for (Operation *pred : hb.predecessorClasses(root)) {
      auto predIt = rootToTimeRange.find(pred);
      if (predIt != rootToTimeRange.end()) {
        // TODO: can do a critical path analysis with argmax
        startTime = std::max(startTime, predIt->second.second);
      }
    }

    rootToTimeRange[root] = {startTime, startTime + duration};
  }
}

TimingAnalysis::TimeRange TimingAnalysis::getTimeRange(Operation *op) const {
  assert(hb && "TimingAnalysis has no backing HappensBeforeAnalysis");
  Operation *root = hb->classRoot(op);
  assert(root && "Operation is not tracked by HappensBeforeAnalysis");
  auto it = rootToTimeRange.find(root);
  assert(it != rootToTimeRange.end() && "Operation class root has no timing entry");
  return it->second;
}

bool TimingAnalysis::isInvalidated(
    const AnalysisManager::PreservedAnalyses &pa) {
  return !pa.isPreserved<TimingAnalysis>() ||
         !pa.isPreserved<HappensBeforeAnalysis>();
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir
