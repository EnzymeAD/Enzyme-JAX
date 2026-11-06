#ifndef PERFIFY_COST_LOOKUP_H
#define PERFIFY_COST_LOOKUP_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace enzyme {
namespace perfify {

template <typename CostMap>
auto findCost(const CostMap &costs, Operation *op) -> decltype(costs.end()) {
  auto it = costs.find(op->getName().getStringRef().str());
  if (it == costs.end())
    it = costs.find((op->getName().getDialectNamespace() + ".*").str());
  if (it == costs.end())
    it = costs.find("*");
  return it;
}

} // namespace perfify
} // namespace enzyme
} // namespace mlir

#endif
