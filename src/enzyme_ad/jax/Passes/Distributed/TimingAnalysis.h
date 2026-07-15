#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_TIMINGANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_TIMINGANALYSIS_H

#include <cstdint>
#include <utility>

#include "src/enzyme_ad/jax/Passes/Distributed/HappensBeforeAnalysis.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace enzyme {
namespace distributed {

class TimingCostModel {
public:
  virtual ~TimingCostModel() = default;
  virtual double getOperationDuration(Operation *op) const = 0;
};

struct AffineTimingCostModelParams {
  double opIntercept = 1.0;
  double kflopCoeff = 0.01;

  double transferIntercept = 1.0;
  double transferKbyteCoeff = 1.0;
};

class AffineTimingCostModel : public TimingCostModel {
public:
  explicit AffineTimingCostModel(
      AffineTimingCostModelParams params = AffineTimingCostModelParams())
      : params(params) {}

  double getOperationDuration(Operation *op) const override;

private:
  AffineTimingCostModelParams params;
};

class TimingAnalysis {
public:
  using TimeRange = std::pair<double, double>;

  // MLIR analysis constructor: depends on HappensBeforeAnalysis and uses
  // an affine duration cost model.
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
