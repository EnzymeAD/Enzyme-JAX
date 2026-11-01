#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SHARDYLOGICALAXISANALYSIS_H

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/IncrementalAxisConstraints.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::distributed {

// Module attribute that switches the axis analysis into Shardy baseline mode.
inline constexpr llvm::StringLiteral kShardyBaselineAttr =
    "distributed.shardy_baseline";

using axis::AxisSymbol;

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

  /**
   * Shardy baseline mode, enabled by the module attribute
   * kShardyBaselineAttr. The expert Shardy plan is then read from the sdy
   * shardings and every symbol Shardy shards is bound to the digit range of
   * the physical mesh it is sharded over, so it is materialized as an
   * axis.factor over a physical axis instead of a fresh logical axis. All
   * other symbols are materialized as device-local axes, since nothing is
   * left for a search to decide.
   */
  bool isShardyBaseline() const { return shardyBaseline; }
  // The physical digit range a (root) symbol is bound to, if Shardy shards it.
  std::optional<PhysicalFactorRef> getPhysicalBinding(AxisSymbol symbol);
  // Builds the axis.factor for a physical digit range over the physical mesh's
  // axes.
  mlir::TypedValue<mlir::enzyme::axis::AxisFactorType>
  createPhysicalFactor(PhysicalFactorRef factor, mlir::OpBuilder &builder,
                       mlir::Location loc);

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
  // Whether a symbol is materialized as a device-local axis, so it never needs
  // communication: one of its resolved roots is local (see
  // IncrementalAxisConstraints::localize), or, in Shardy baseline mode, Shardy
  // does not shard it over any physical digits.
  bool isUnshardable(AxisSymbol symbol);
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
  // Input (LHS) and result (RHS) symbols of each resharding op (see
  // getReshardingOp), one per dimension. A dimension the op may reshard gets
  // distinct symbols on its two sides; any other shares one.
  llvm::DenseMap<Operation *, DimToSymbol> reshardLHSSymbols;
  llvm::DenseMap<Operation *, DimToSymbol> reshardRHSSymbols;
  // Explicit view casts assign logical axes to serialized SSA factors.
  // These maps must agree after all symbol factoring and unification.
  llvm::DenseMap<Value, AxisSymbol> factorToLogicalAxis;
  llvm::DenseMap<AxisSymbol, Value> logicalAxisToFactor;
  axis::IncrementalAxisConstraints axisConstraints;
  Operation *sdy_func = nullptr;
  bool valid = true;

  // Shardy baseline state (see isShardyBaseline).
  //
  // An atom is a symbol created for one physical digit range, either by
  // seeding a dimension from its Shardy sharding or by anchoring a frozen
  // physical factor. Merges may later factor an atom into several roots, so
  // per-root bindings are derived from the atoms after all unions.
  struct BoundAtom {
    AxisSymbol symbol;
    PhysicalFactorRef factor;
    Operation *diagOp;
    std::string description;
  };
  // One tensor dimension seeded from Shardy: `symbols` is the sharded atoms
  // major-first followed by the unsharded remainder, and `original` is the
  // symbol list the dimension had before seeding.
  struct DimSeed {
    Operation *diagOp;
    std::string description;
    llvm::SmallVector<AxisSymbol> original;
    llvm::SmallVector<AxisSymbol> symbols;
    llvm::SmallVector<PhysicalFactorRef> expected;
  };
  bool shardyBaseline = false;
  ModuleOp baselineModule;
  PhysicalMeshOp baselinePhysicalMesh;
  llvm::SmallVector<PhysicalAxisSpec> physicalAxes;
  llvm::SmallVector<Value> physicalAxisValues;
  llvm::SmallVector<BoundAtom> boundAtoms;
  llvm::SmallVector<DimSeed> dimSeeds;
  llvm::DenseMap<AxisSymbol, PhysicalFactorRef> rootBindings;
  bool rootBindingsDirty = true;
  void initShardyBaseline();
  // Merges each dimension's symbols with the factorization implied by its
  // Shardy sharding, before propagation unions run, so propagation cannot
  // move a shard boundary.
  void seedShardyShardings();
  void seedTensorDims(Operation *diagOp, llvm::StringRef what, Value value,
                      mlir::sdy::TensorShardingAttr sharding,
                      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> dims);
  // The physical digit ranges, major-first, each dimension of a Shardy
  // sharding is sharded over.
  FailureOr<llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>>>
  projectShardingDims(Operation *diagOp, llvm::StringRef what,
                      mlir::sdy::TensorShardingAttr sharding);
  // Merges one producer-to-consumer edge dimension by dimension, except where
  // Shardy leaves the two sides deliberately different. Shardy does not insert
  // a reshard when an op shards a dimension further than its operand is
  // (each device slices its operand locally), nor across a call boundary; such
  // an edge keeps separate axes so materialization sees a collective.
  void
  mergeEdgeRespectingShardy(const TensorAxesToPartitionAxes &producerMapping,
                            Value producerValue,
                            const TensorAxesToPartitionAxes &consumerMapping,
                            OpOperand &use, bool isBoundary);
  // Projection of a sharding onto the physical mesh, memoized.
  const llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>> *
  getProjection(Operation *diagOp, llvm::StringRef what,
                mlir::sdy::TensorShardingAttr sharding);
  llvm::DenseMap<mlir::Attribute,
                 llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>>>
      projectionCache;
  // Recomputes the physical digit range of every root symbol. Conflicts are
  // reported only when `diagnose` is set, which is for the final check.
  void rebuildRootBindings(bool diagnose);
  // Checks that the unions kept every Shardy sharding intact.
  void validateShardyBaseline();
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
  TensorAxesToPartitionAxes getTensorPartitionDims(ReshardingOp reshard,
                                                   bool isLHS);
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

// Creates a new axis for `symbol` and returns it viewed as a factor: a
// DeviceLocalAxisOp if `axisAnalysis` says the symbol must never be sharded
// across the mesh, otherwise a LogicalMeshAxesOp. In Shardy baseline mode a
// symbol bound to a physical digit range becomes a factor over the physical
// axis instead, and every other symbol a DeviceLocalAxisOp. Device-local axes
// stay in a tensor's axes so every factor keeps its position within its
// dimension.
//
// Always creates a new op. Callers own their symbol-to-factor cache (and any
// reuse of axes the wider analysis already knows about) and call this only on
// a cache miss.
mlir::TypedValue<mlir::enzyme::axis::AxisFactorType>
createLogicalAxisFactor(AxisSymbol symbol,
                        ShardyLogicalAxisAnalysis &axisAnalysis,
                        mlir::OpBuilder &builder, mlir::Location loc);

} // namespace mlir::enzyme::distributed

#endif
