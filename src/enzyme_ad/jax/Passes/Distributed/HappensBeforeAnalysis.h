#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_HAPPENSBEFOREANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_HAPPENSBEFOREANALYSIS_H

#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir {
namespace enzyme {
namespace distributed {

/// Analyzes happens-before relationships for a given operation.
///
/// This analysis tracks immediate happens-before edges between simultaneous
/// equivalence classes (not individual members).
class HappensBeforeAnalysis {
public:
  HappensBeforeAnalysis(Operation *op);

  // Query methods
  /// Returns true if `a`'s class is an immediate predecessor of `b`'s class.
  bool happensBefore(Operation *a, Operation *b) const;
  bool simultaneousWith(Operation *a, Operation *b) const;

  /// Returns all members of the class rooted at `classRoot`.
  llvm::SmallVector<Operation *> classList(Operation *classRoot) const;
  /// Returns the root of `classMember`'s equivalence class.
  Operation *classRoot(Operation *classMember) const;

  /// Returns predecessor classes for `a`'s class.
  llvm::SmallVector<Operation *> predecessorClasses(Operation *a) const;
  /// Returns successor classes for `a`'s class.
  llvm::SmallVector<Operation *> successorClasses(Operation *a) const;

  /// Returns all class roots in a valid topological order (predecessors before
  /// successors). Computed once at construction; if a cycle exists the
  /// constructor asserts (deadlock detection).
  const llvm::SmallVector<Operation *> &classesInTopologicalOrder() const;

  // Optional: Control analysis invalidation
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);

private:
  llvm::EquivalenceClasses<Operation *> simultaneousClasses;
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> predecessorsMap;
  // should be transpose of predecessorsMap, for lookups in the other direction
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> successorsMap;
  
  void markSimultaneous(Operation *a, Operation *b);
  void scanSimultaneousOperations(LaneOpInterface laneOp);

  void addHappensBeforeEdge(Operation *a, Operation *b);
  void scanHappensBeforeEdges(LaneOpInterface laneOp);
  void computeTopologicalOrder();

  llvm::SmallVector<Operation *> topologicalOrder;
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_HAPPENSBEFOREANALYSIS_H
