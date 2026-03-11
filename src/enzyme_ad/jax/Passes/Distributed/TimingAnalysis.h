#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_TIMINGANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_TIMINGANALYSIS_H

#include <cstdint>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "src/enzyme_ad/jax/Passes/Distributed/HappensBeforeAnalysis.h"

namespace mlir {
namespace enzyme {
namespace distributed {

class TimingCostModel {
public:
  virtual ~TimingCostModel() = default;
  virtual int64_t getOperationDuration(Operation *op) const = 0;
};

class UnitTimingCostModel : public TimingCostModel {
public:
  int64_t getOperationDuration(Operation *op) const override;
};

class TimingAnalysis {
public:
  using TimeRange = std::pair<int64_t, int64_t>;

  // MLIR analysis constructor: depends on HappensBeforeAnalysis and uses
  // a unit-duration fallback cost model.
  TimingAnalysis(Operation *op, AnalysisManager &am);

  // Explicit constructor for custom cost models.
  TimingAnalysis(const HappensBeforeAnalysis &hb,
                 const TimingCostModel &costModel);

  TimeRange getTimeRange(Operation *op) const;

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);

private:
  const HappensBeforeAnalysis *hb = nullptr;
  llvm::DenseMap<Operation *, TimeRange> rootToTimeRange;

  void buildTimingMap(const HappensBeforeAnalysis &hb,
                      const TimingCostModel &costModel);
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_TIMINGANALYSIS_H
