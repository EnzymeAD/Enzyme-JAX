#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H

#include <atomic>
#include <cassert>
#include <cstdint>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

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
  using OverlapSet = llvm::DenseSet<AxisSymbol>;

  void mergeSymbols(AxisSymbol a, AxisSymbol b) {
    attemptMergeSymbols({a}, {b});
  }
  void mergeSymbols(llvm::ArrayRef<AxisSymbol> a,
                    llvm::ArrayRef<AxisSymbol> b) {
    attemptMergeSymbols(a, b);
  }
  /**
   * Marks a set of symbols as "overlapping", which should never
   * be merged. Symmetric, antireflexive, but not transitive.
   */
  void markOverlapping(llvm::ArrayRef<AxisSymbol> overlapping);
  /**
   * Resolves the root factorization of a symbol or symbol
   * list after traversing unions and merges.
   */
  llvm::SmallVector<AxisSymbol> resolve(AxisSymbol sym);
  llvm::SmallVector<AxisSymbol> resolve(llvm::ArrayRef<AxisSymbol> syms);
  // Promises to return a set of root symbols
  OverlapSet getOverlapping(AxisSymbol sym);
  OverlapSet getOverlapping(llvm::ArrayRef<AxisSymbol> syms);

private:
  llvm::EquivalenceClasses<AxisSymbol> symbolUnion;
  llvm::DenseMap<AxisSymbol, llvm::SmallVector<AxisSymbol>> factorizations;
  llvm::DenseMap<AxisSymbol, OverlapSet> overlappingSymbols;
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
  void _factorSymbol(AxisSymbol symbol, llvm::ArrayRef<AxisSymbol> factors);
  void _mergeSymbols(AxisSymbol a, AxisSymbol b);
  void _appendOverlaps(const OverlapSet &from, AxisSymbol to);
  void _resolve(AxisSymbol sym, llvm::SmallVector<AxisSymbol> &result);
  OverlapSet _getOverlappingForRoot(AxisSymbol sym);
  // we never look at the overlapping for non-root,
  // so whenver we merge or factor we can clear out entry.
  // Also helps make it more obvious when we look at a non-root.
  void _clearOverlapping(AxisSymbol sym) { overlappingSymbols.erase(sym); }
};

/**
 * Attempts to find a basis of logical axes where no operands / values
 * use the same logical partitioning axis unless there is a true dependency
 * between them- either directly (eventual producer sharding --> consumer
 * sharding) relation or indirectly (i.e. both eventually are consumed by an op
 * that shards both inputs along an axis). Respects any existing reshardings
 * rules that "break" a propagation dependency.
 *
 * Results in a mapping from the following IR items to logical axes:
 *  - ops to logical axes: for each op, provides a logical axis for each axis of
 * its shardy partitioning rule.
 *  - op result to logical axes: the sharding that the producer op would have,
 * based on its own sharding. Maps tensor axes to partitioning axes.
 *  - op use / operand to logical axes: the sharding that the consuming op would
 * have, based on its own sharding. If this differs from the producer's
 * sharding, then a resharding is needed.
 *  - block args to logical axes: this will attempt to unify the logical axes of
 * the block args to the consumer partitioning. TBD: what happens when more than
 * one consumer has a different partitioning?
 */
class ShardyLogicalAxisAnalysis {
public:
  ShardyLogicalAxisAnalysis(Operation *sdy_func);
  ShardyLogicalAxisAnalysis() = default;

  bool isValid() const { return valid; }

  using SymbolsPerPartitioningAxis =
      llvm::SmallVector<llvm::SmallVector<AxisSymbol>>;
  using TensorAxesToPartitionAxes =
      llvm::SmallVector<llvm::SmallVector<AxisSymbol>>;
  using BlockArgumentToPartitionAxes =
      llvm::DenseMap<BlockArgument, TensorAxesToPartitionAxes>;

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
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDims(BlockArgument arg);
  // Associates a resolved logical axis with its canonical serialized factor.
  // Distinct factors may not be assigned to the same resolved logical axis.
  LogicalResult assignLogicalAxis(AxisSymbol symbol, Value factor);
  // Returns the factor previously assigned to a resolved logical axis.
  Value getLogicalAxis(AxisSymbol symbol) const;
  // Redirects analysis bookkeeping from one op to another after a rewrite.
  void markRewrite(Operation *from, Operation *to);
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
  BlockArgumentToPartitionAxes argToPartitioningAxes;
  llvm::DenseMap<Operation *, DimToSymbol> reshardLHSSymbols;
  llvm::DenseMap<Operation *, DimToSymbol> reshardRHSSymbols;
  // Explicit view casts assign logical axes to serialized SSA factors.
  // These maps must agree after all symbol factoring and unification.
  llvm::DenseMap<Value, AxisSymbol> factorToLogicalAxis;
  llvm::DenseMap<AxisSymbol, Value> logicalAxisToFactor;
  SymbolFactorMerge symbolFactorMerge;
  Operation *sdy_func = nullptr;
  bool valid = true;

  void buildInitialSymbols();
  void validateLogicalAxisAssignments();
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
  getTensorPartitionDimsForViewCast(ValueRange partitioningAxes);
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDims(Operation *op, bool isLHS, int valueIdx);
  void buildUnion();
};

// Module-scoped wrapper that materializes ShardyLogicalAxisAnalysis for
// func.func @main when available. Essentially a wrapper allowing
// for getAnalysis<>() to be called on a module.
class MainFunctionShardyLogicalAxisAnalysis {
public:
  MainFunctionShardyLogicalAxisAnalysis(ModuleOp module,
                                        AnalysisManager &analysisManager);

  bool isValid() const { return valid; }

  ShardyLogicalAxisAnalysis &getAnalysis() {
    assert(isValid() && "main function axis analysis unavailable");
    return *analysis;
  }
  const ShardyLogicalAxisAnalysis &getAnalysis() const {
    assert(isValid() && "main function axis analysis unavailable");
    return *analysis;
  }

private:
  bool valid = true;
  ShardyLogicalAxisAnalysis *analysis = nullptr;
};

// Shared debug-dump helpers, usable from any pass in the distributed
// pipeline. Both walk `block` linearly (no recursion into nested regions)
// and print even when the analysis or surrounding IR is only partially
// valid, since their whole purpose is diagnosing broken invariants.
//
// Prints, for every block argument and op result in `block`, its producer-side
// logical partitioning axes as one bracketed list per tensor dimension, e.g.
// `%val : [%ax1, %ax2] [%ax3] [%ax4]`. Values with no tracked sharding (e.g.
// non-tensor values) are skipped.
void dumpValueAxes(llvm::raw_ostream &os, Block *block,
                   ShardyLogicalAxisAnalysis &axisAnalysis);

// Prints, for every op in `block`, its Shardy rewrite partitioning axes as one
// bracketed list per sharding-rule factor, e.g. `some.op : [%ax1] [%ax2,
// %ax3]`.
void dumpOperationAxes(llvm::raw_ostream &os, Block *block,
                       ShardyLogicalAxisAnalysis &axisAnalysis);

} // namespace mlir::enzyme::distributed

#endif