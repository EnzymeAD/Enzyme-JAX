#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H

#include <atomic>
#include <cassert>
#include <cstdint>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
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
   * Marks a set of symbols as "unshardable": they still participate
   * normally in union-find merging (this never blocks a merge the way
   * markOverlapping does), but must never be sharded across the mesh, so
   * they are materialized as device-local axes. Used for factors Shardy
   * tags as needing a collective-permute if sharded (e.g. the sliced-away
   * dimension of a static per-layer stablehlo.slice) -- we don't implement
   * collective-permute, so we instead guarantee these axes are never
   * actually sharded at all. Symbols anchored to a device-local axis in the
   * IR are tagged the same way.
   */
  void markUnshardable(llvm::ArrayRef<AxisSymbol> symbols);
  // Whether a symbol's resolved root has been tagged unshardable, directly
  // or via a merge with an unshardable symbol.
  bool isUnshardable(AxisSymbol sym);
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
  llvm::DenseSet<AxisSymbol>
      unshardableSymbols; // keyed by root, like overlappingSymbols
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
 * The analysis covers the function it is constructed on plus every function
 * reachable from it through func.call. All of them share one symbol space, and
 * a call is transparent: its operand k is consumed with the callee's argument-k
 * symbols, and its result k is produced with the callee's return-operand-k
 * symbols. Every call site of one callee therefore unifies on the callee's
 * argument and result axes, so repeated structure factored into a function
 * contributes one set of degrees of freedom rather than one per call site.
 * The call graph must be acyclic (recursive calls are not supported).
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

  // The analyzed function-like ops: sdy_func first, then its transitive
  // callees.
  llvm::ArrayRef<Operation *> getAnalyzedFunctions() const {
    return analyzedFuncs;
  }
  /**
   * Returns the set of symbols that an op result
   * needs to be reduced over to produce the correct
   * global result.
   */
  llvm::SmallVector<AxisSymbol> getReductionAxes(OpResult result);
  // Whether a symbol's resolved root has been tagged unshardable (see
  // SymbolFactorMerge::markUnshardable).
  bool isUnshardable(AxisSymbol symbol) {
    return symbolFactorMerge.isUnshardable(symbol);
  }
  /**
   * Returns a copy of a tensor's per-dimension partitioning-axis lists with
   * every unshardable symbol dropped. Unshardable symbols stay in the axes a
   * tensor is materialized with, as device-local axes, so this is only for
   * deciding whether communication is needed: unshardable axes never need a
   * collective, so they must not count as a sharding conflict or a reduction.
   */
  TensorAxesToPartitionAxes
  excludeUnshardable(const TensorAxesToPartitionAxes &axes);
  // Flat-list overload for callers like getReductionAxes's result that
  // aren't shaped per-dimension.
  llvm::SmallVector<AxisSymbol>
  excludeUnshardable(llvm::ArrayRef<AxisSymbol> symbols);

private:
  // two loops: one vector over dimensions, one vector over
  // symbols within those dimensions.

  using DimToSymbol = llvm::SmallVector<AxisSymbol, 4>;
  llvm::DenseMap<Operation *, DimToSymbol> opToPartitioningAxes;
  // For an op with permutation or need-replication factors, the symbol each
  // operand / result gets for every dimension one of them spans (null for other
  // dimensions), keyed by (op, operand or result number). Such a factor is
  // sized by the op's own extent for it, which differs between tensors (the
  // sliced-away axis of a slice is 2 on its operand and 1 on its result), and
  // nothing is sharded along it, so each tensor gets an unshardable symbol of
  // its own dimension's size instead of sharing the factor's.
  using TensorLocalSymbols =
      llvm::DenseMap<std::pair<Operation *, int64_t>, DimToSymbol>;
  TensorLocalSymbols operandLocalSymbols;
  TensorLocalSymbols resultLocalSymbols;
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
#ifndef NDEBUG
  // getReductionAxes's debug-only informational remarks, emitted only once
  // each: an unrolled model repeats the same permutation/need-replication
  // classification across many ops, and a single remark says everything the
  // per-occurrence spam would.
  bool emittedPermutationFactorRemark = false;
  bool emittedNeedReplicationFactorRemark = false;
#endif

  // Functions whose bodies participate in the analysis: sdy_func first, then
  // its transitive callees.
  llvm::SmallVector<Operation *> analyzedFuncs;
  void collectAnalyzedFunctions();
  void buildTensorLocalSymbols(Operation *op,
                               mlir::sdy::OpShardingRuleAttr shardingRule);
  void buildInitialSymbolsFor(Operation *func);
  // The symbol anchored to a frozen SSA axis factor, created on first use.
  std::optional<AxisSymbol>
  getOrCreateSymbolForFactor(mlir::TypedValue<axis::AxisFactorType> factor);
  // A converted function argument's axes, recovered from the function's own
  // sharding metadata. Empty for a function that is not yet converted.
  std::optional<TensorAxesToPartitionAxes>
  getFrozenArgumentAxes(Operation *func, mlir::BlockArgument arg);
  // Merges the producer-to-consumer edges within `func`'s body. Call
  // boundary edges (operand to call, call result to its uses) are handled
  // only when `boundaryEdges` is true, and all other edges only when false.
  void buildUnionFor(Operation *func, bool boundaryEdges);
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
  // Mapping for a view cast or anchor `op`, read from its explicit
  // partitioning_axes operands.
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDimsFromPartitioningAxes(Operation *op);
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDims(Operation *op, bool isLHS, int valueIdx);
  // Callee-side view of a func.call: the argument (consumer) or return operand
  // (producer) whose symbols the call boundary shares.
  std::optional<TensorAxesToPartitionAxes>
  getTensorPartitionDimsForCall(CallOpInterface call, bool isLHS, int valueIdx);
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

// The per-device type of a global tensor: each dimension divided by the
// product of its partitioning axes' extents.
mlir::RankedTensorType getLocalTensorType(
    mlir::RankedTensorType globalType,
    llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes);

} // namespace mlir::enzyme::distributed

#endif
