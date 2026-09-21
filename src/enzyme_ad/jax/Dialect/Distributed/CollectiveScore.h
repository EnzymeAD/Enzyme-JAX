#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_SCORE_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_SCORE_H

#include "CollectiveDecomposer.h"

#include "mlir/IR/BuiltinOps.h"

#include <map>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {

// The communication cost of a lowered module, as the sum of its collectives'
// isolated durations.
//
// The model is deliberately simple:
//   - Each atomic distributed.Collective is decomposed by the collective
//     planner and costed alone, at the uniform MeshCostParams of the module's
//     physical mesh.
//   - Durations are added. Collectives that could run concurrently are not
//     overlapped, and computation is not costed.
struct CollectiveCostSummary {
  // Sum of `durations`. Meaningful only when `feasible`.
  double total = 0;
  // One entry per collective, in module walk order.
  std::vector<double> durations;
  // Empty when every collective could be costed; otherwise why the first one
  // that could not was rejected.
  std::string failureReason;
  // Number of collectives in the module, including any that failed.
  size_t numCollectives = 0;

  bool feasible() const { return failureReason.empty(); }
};

// Costs modules produced by the same search, memoizing plan results. Plans are
// exact (default PlanOptions).
//
// Collectives that normalize to the same NormalizedCollective under the same
// mesh parameters have the same plan, and a lowered program repeats them (one
// per layer, per candidate), so results are cached by that value.
class CollectiveCostModel {
public:
  // Costs every distributed.Collective in `module`, which must have run the
  // lowering pipeline's atomize-collectives step and contain a unique
  // physical mesh.
  CollectiveCostSummary summarize(ModuleOp module);

  size_t cacheHits() const { return hits; }
  size_t cacheMisses() const { return misses; }

private:
  struct Entry {
    bool ok = false;
    double duration = 0;
    std::string failureReason;
  };

  std::map<std::string, Entry> cache;
  size_t hits = 0;
  size_t misses = 0;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_SCORE_H
