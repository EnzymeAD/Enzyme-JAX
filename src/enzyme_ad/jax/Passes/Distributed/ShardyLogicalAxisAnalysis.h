#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H

#include <atomic>
#include <cassert>
#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::enzyme::distributed {

// Unique counter object.
class AxisSymbol {
public:
  AxisSymbol() = default;
  AxisSymbol(const AxisSymbol &) = default;
  AxisSymbol &operator=(const AxisSymbol &) = default;

  static AxisSymbol create(uint64_t extent) {
    return AxisSymbol(nextId.fetch_add(1, std::memory_order_relaxed), extent);
  }

  static AxisSymbol getEmptyKey() { return AxisSymbol(kEmptyId, 0); }
  static AxisSymbol getTombstoneKey() { return AxisSymbol(kTombstoneId, 0); }

  bool isNull() const { return id == kNullId; }
  uint64_t getId() const { return id; }
  uint64_t getExtent() const { return extent; }
  bool operator==(const AxisSymbol &other) const { return id == other.id; }

private:
  explicit AxisSymbol(uint64_t id, uint64_t extent) : id(id), extent(extent) {}

  static constexpr uint64_t kNullId = 0;
  static constexpr uint64_t kEmptyId = UINT64_MAX;
  static constexpr uint64_t kTombstoneId = UINT64_MAX - 1;
  inline static std::atomic<uint64_t> nextId{1};
  uint64_t id = kNullId;
  uint64_t extent;
};

} // namespace mlir::enzyme::distributed

namespace llvm {

template <> struct DenseMapInfo<mlir::enzyme::distributed::AxisSymbol> {
  static mlir::enzyme::distributed::AxisSymbol getEmptyKey() {
    return mlir::enzyme::distributed::AxisSymbol::getEmptyKey();
  }

  static mlir::enzyme::distributed::AxisSymbol getTombstoneKey() {
    return mlir::enzyme::distributed::AxisSymbol::getTombstoneKey();
  }

  static unsigned getHashValue(const mlir::enzyme::distributed::AxisSymbol &k) {
    return DenseMapInfo<uint64_t>::getHashValue(k.getId());
  }

  static bool isEqual(const mlir::enzyme::distributed::AxisSymbol &lhs,
                      const mlir::enzyme::distributed::AxisSymbol &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace mlir::enzyme::distributed {

class SymbolFactorMerge {
public:
  void mergeSymbols(AxisSymbol a, AxisSymbol b) {
    attemptMergeSymbols({a}, {b});
  }
  void mergeSymbols(llvm::ArrayRef<AxisSymbol> a,
                    llvm::ArrayRef<AxisSymbol> b) {
    attemptMergeSymbols(a, b);
  }
  /**
   * Resolves the root factorization of a symbol or symbol
   * list after traversing unions and merges.
   */
  llvm::SmallVector<AxisSymbol> resolve(AxisSymbol sym);
  llvm::SmallVector<AxisSymbol> resolve(llvm::ArrayRef<AxisSymbol> syms);

private:
  llvm::EquivalenceClasses<AxisSymbol> symbolUnion;
  llvm::DenseMap<AxisSymbol, llvm::SmallVector<AxisSymbol>> factorizations;
  /**
   * Between two lists of (possibly factored) symbols,
   * performs as much of a merge between factors as possible.
   * Merges may not be possible due to indivisibility: for instance,
   * [3, 2] and [2, 3] cannot be merged.
   *
   * Expects that each list is unique even after resolution
   * to root factors. If not, the behavior is undefined.
   */
  void attemptMergeSymbols(llvm::ArrayRef<AxisSymbol> a,
                           llvm::ArrayRef<AxisSymbol> b);
  void _resolve(AxisSymbol sym, llvm::SmallVector<AxisSymbol> &result);
};

class ShardyLogicalAxisAnalysis {
public:
  ShardyLogicalAxisAnalysis(func::FuncOp sdy_func);
  ShardyLogicalAxisAnalysis() = default;

  using SymbolsPerPartitioningAxis =
      llvm::SmallVector<llvm::SmallVector<AxisSymbol>>;
  using TensorAxesToPartitionAxes =
      llvm::SmallVector<llvm::SmallVector<AxisSymbol>>;

  /**
   * Returns the set of symbols we have assigned to each
   * Shardy rewrite partitioning axes. The list is given majormost
   * first.
   */
  SymbolsPerPartitioningAxis getPartitioningAxes(Operation *op);
  /**
   * Returns the set of dimensions that each tensor axis is
   * sharded over, from the perspective of the producer.
   * (This will not match the consumer only in cases needing
   * a reshard or collective)
   */
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDims(OpResult result);
  /**
   * Returns the set of dimensions that each tensor axis is
   * sharded over, from the perspective of the consumer.
   * (This will not match the producer only in cases needing
   * a reshard or collective)
   */
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDims(OpOperand &use);
  /**
   * Returns the set of symbols that an op result
   * needs to be reduced over to produce the correct
   * global result.
   */
  llvm::SmallVector<AxisSymbol> getReductionAxes(OpResult result);

private:
  // two loops: one vector over dimensions, one vector over
  // symbols within those dimensions.

  using DimToSymbol = llvm::SmallVector<AxisSymbol, 4>;
  llvm::DenseMap<Operation *, DimToSymbol> opToPartitioningAxes;
  llvm::DenseMap<Operation *, DimToSymbol> reshardLHSSymbols;
  llvm::DenseMap<Operation *, DimToSymbol> reshardRHSSymbols;
  SymbolFactorMerge symbolFactorMerge;
  func::FuncOp sdy_func;

  void buildInitialSymbols();
  // Internal implementation for either a producer (lhs) or consumer (rhs) of a
  // tensor. Three versions: one for a generic op, which may or may not
  // be able to find a sharding rule, and two specializations.
  TensorAxesToPartitionAxes
  getTensorPartitionDims(Operation *op,
                         mlir::sdy::OpShardingRuleAttr shardingRule, bool isLHS,
                         int valueIdx);
  TensorAxesToPartitionAxes getTensorPartitionDims(mlir::sdy::ReshardOp op,
                                                   bool isLHS, int valueIdx);
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDims(Operation *op, bool isLHS, int valueIdx);
  void buildUnion();
};

} // namespace mlir::enzyme::distributed

#endif