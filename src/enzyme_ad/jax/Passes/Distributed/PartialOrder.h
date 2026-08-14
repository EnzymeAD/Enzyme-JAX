

#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_PARTIALORDER_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_PARTIALORDER_H

#include <cassert>

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::enzyme::distributed {

enum class Order { LessThan, Equal, GreaterThan, Incomparable };

/**
 * A build-then-query data structure for a partial order.
 * Not necessarly the most efficient: at least N^2 memory.
 * After building the transitive closure, lookups should be
 * relatively cheap (lookup within O(N) size set)
 */
template <typename KeyType> class PartialOrder {
private:
  llvm::DenseMap<KeyType, llvm::DenseSet<KeyType>> direct_edges;
  llvm::DenseMap<KeyType, llvm::DenseSet<KeyType>> transitive_edges;
  bool closed;
  int node_count = 0;

public:
  PartialOrder() : closed(false) {}

  /**
   * This can only be called after closing the partial order
   */
  Order compare(const KeyType lhs, const KeyType rhs);
  /**
   * Adding cycles is not allowed!
   */
  void addOrder(const KeyType lesser, const KeyType greater);
  void close();
};

template <typename KeyType>
Order PartialOrder<KeyType>::compare(const KeyType lhs, const KeyType rhs) {
  assert(closed && "Partial order must be closed before comparing");
  if (lhs == rhs) {
    return Order::Equal;
  }
  if (transitive_edges[lhs].count(rhs)) {
    return Order::LessThan;
  }
  if (transitive_edges[rhs].count(lhs)) {
    return Order::GreaterThan;
  }
  return Order::Incomparable;
}

template <typename KeyType>
void PartialOrder<KeyType>::addOrder(const KeyType lesser,
                                     const KeyType greater) {
  assert(!closed && "Cannot add order after closing the partial order");
  if (!direct_edges.count(lesser)) {
    node_count++;
  }
  if (!direct_edges.count(greater)) {
    node_count++;
    direct_edges[greater] = llvm::DenseSet<KeyType>();
  }
  direct_edges[lesser].insert(greater);
}

template <typename KeyType> void PartialOrder<KeyType>::close() {
  assert(!closed && "Cannot close an already closed partial order");
  closed = true;

  // In an acyclic graph, we can follow to the end of the graph recursively
  // so that each downstream set is complete before we add it to the lesser
  // node's set. As a cycle defence, check depth is less than N
  auto dfs = [&](const KeyType node, int depth, auto &&dfs_ref) -> void {
    assert(depth < node_count && "Cycle detected in partial order");
    for (const auto &child : direct_edges[node]) {
      dfs_ref(child, depth + 1, dfs_ref);
      transitive_edges[node].insert(child);
      transitive_edges[node].insert(transitive_edges[child].begin(),
                                    transitive_edges[child].end());
    }
    // To avoid revisiting the same child, we can clear the direct edges after
    // processing
    direct_edges[node].clear();
  };
  for (const auto &node : direct_edges) {
    dfs(node.first, 0, dfs);
  }
}

/**
 * Gives a partial order for a single block, where dependency is defined solely
 * by SSA use-def chains.
 */
class SSABlockPartialOrderAnalysis {
public:
  SSABlockPartialOrderAnalysis(mlir::Block *block) : block(block) {
    if (!block) {
      valid = false;
      return;
    }
    buildPartialOrder();
  }
  SSABlockPartialOrderAnalysis(Operation *op) {
    if (!op || op->getNumRegions() == 0 || !op->getRegion(0).hasOneBlock()) {
      valid = false;
      return;
    }
    block = &op->getRegion(0).front();
    buildPartialOrder();
  }

  bool isValid() const { return valid; }

  PartialOrder<mlir::Operation *> &getPartialOrder() { return partial_order; }
  const PartialOrder<mlir::Operation *> &getPartialOrder() const {
    return partial_order;
  }

private:
  bool valid = true;
  mlir::Block *block;
  void buildPartialOrder();

  PartialOrder<mlir::Operation *> partial_order;
};

// Module-scoped wrapper that computes SSA block partial order for @main
// entry block when available.
class MainFunctionSSABlockPartialOrderAnalysis {
public:
  MainFunctionSSABlockPartialOrderAnalysis(ModuleOp module,
                                           AnalysisManager &analysisManager);

  bool isValid() const { return valid; }

  SSABlockPartialOrderAnalysis &getAnalysis() {
    assert(valid && analysis && "main block partial order unavailable");
    return *analysis;
  }
  const SSABlockPartialOrderAnalysis &getAnalysis() const {
    assert(valid && analysis && "main block partial order unavailable");
    return *analysis;
  }

  PartialOrder<mlir::Operation *> &getPartialOrder() {
    return getAnalysis().getPartialOrder();
  }
  const PartialOrder<mlir::Operation *> &getPartialOrder() const {
    return getAnalysis().getPartialOrder();
  }

private:
  bool valid = true;
  SSABlockPartialOrderAnalysis *analysis = nullptr;
};
} // namespace mlir::enzyme::distributed

#endif