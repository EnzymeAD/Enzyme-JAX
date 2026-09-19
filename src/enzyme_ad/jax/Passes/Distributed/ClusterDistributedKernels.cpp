#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <utility>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_CLUSTERDISTRIBUTEDKERNELSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

static void
dumpLogicalAxesForMainBlock(Block *mainBlock,
                            ShardyLogicalAxisAnalysis &axisAnalysis) {
  llvm::errs() << "[ClusterDistributedKernels] logical axes for main block\n";
  for (BlockArgument arg : mainBlock->getArguments()) {
    auto partitioningAxes = axisAnalysis.getTensorPartitionDims(arg);
    llvm::errs() << "  arg: " << arg << "\n";
    llvm::errs() << "    partitioning axes: ";
    for (const auto &axis : *partitioningAxes) {
      llvm::errs() << "[";
      for (const auto &symbol : axis) {
        llvm::errs() << "a" << symbol.getId() << ":" << symbol.getExtent()
                     << " ";
      }
      llvm::errs() << "] ";
    }
    llvm::errs() << "\n";
  }
  for (Operation &op : mainBlock->getOperations()) {
    auto partitioningAxes = axisAnalysis.getPartitioningAxes(&op);

    llvm::errs() << "  op: " << op << "\n";
    llvm::errs() << "    partitioning axes: ";
    for (const auto &axis : partitioningAxes) {
      llvm::errs() << "[";
      for (const auto &symbol : axis) {
        llvm::errs() << "a" << symbol.getId() << ":" << symbol.getExtent()
                     << " ";
      }
      llvm::errs() << "] ";
    }
    llvm::errs() << "\n";
  }
}

mlir::RankedTensorType
toLocalType(mlir::RankedTensorType globalType,
            llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
  auto globalShape = globalType.getShape();
  assert(globalShape.size() == partitioningAxes.size() &&
         "global shape and partitioning axes must have the same rank");
  llvm::SmallVector<int64_t> localShape;
  localShape.reserve(globalShape.size());
  for (size_t i = 0; i < globalShape.size(); ++i) {
    int64_t globalDim = globalShape[i];
    uint64_t extent = 1;
    for (const auto &symbol : partitioningAxes[i]) {
      extent *= symbol.getExtent();
    }
    assert(globalDim % extent == 0 &&
           "global dimension must be divisible by partitioning extent");
    localShape.push_back(globalDim / extent);
  }
  return mlir::RankedTensorType::get(localShape, globalType.getElementType());
}

using TensorPartitioningAxes =
    ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis;

static Type getLocalTypeForValue(
    Value value,
    llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
  auto rankedType = dyn_cast<RankedTensorType>(value.getType());
  if (!rankedType) {
    return value.getType();
  }
  return toLocalType(rankedType, partitioningAxes);
}

static std::optional<TensorPartitioningAxes>
getPartitioningForValue(Value value, ShardyLogicalAxisAnalysis &axisAnalysis) {
  if (auto result = dyn_cast<OpResult>(value)) {
    return axisAnalysis.getTensorPartitionDims(result);
  }
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    return axisAnalysis.getTensorPartitionDims(arg);
  }
  return std::nullopt;
}

// Historically this walked through a neighboring
// builtin.unrealized_conversion_cast when direct analysis failed. That fallback
// is gone: ShardyLogicalAxisAnalysis now dispatches directly through
// DistributedCastGlobalToLocalOp, CastLocalToGlobalOp, and AnchorPartitioningOp
// (their partitioning_axes are ground truth), so a value produced by any of
// those already resolves via getPartitioningForValue itself -- no cast-shaped
// op is opaque to the analysis anymore.
static std::optional<TensorPartitioningAxes>
getPartitioningForValueOrCastNeighborhood(
    Value value, ShardyLogicalAxisAnalysis &axisAnalysis) {
  return getPartitioningForValue(value, axisAnalysis);
}

// Determines whether `value` is already at the distributed dialect's LOCAL
// (per-device shard) scope as opposed to GLOBAL (full-tensor) scope, by
// walking back through scope-preserving ops. This can no longer be read off
// a value's own tensor shape: AnchorPartitioningOp is scope-agnostic (its
// AllTypesMatch input/output are the identical type, whichever scope that
// happens to be), so the scope has to be established by whatever real
// boundary sits further upstream -- a Cast op, a collective's own Await
// (always local -- see this file's DistributedCollectiveOp/DistributedAwait
// usage), or a DistributedKernelOp's own operand/result (always local).
// Anything else (a plain compute op's result, a function block argument) is
// GLOBAL by default, matching this dialect's baseline assumption that
// ordinary tensor values are full-tensor until something explicitly shards
// them.
static bool isLocallyScopedValue(Value value) {
  for (int step = 0; step < 8; ++step) {
    Operation *producer = value.getDefiningOp();
    if (!producer) {
      return false;
    }
    if (isa<DistributedCastGlobalToLocalOp>(producer)) {
      return true;
    }
    if (isa<DistributedCastLocalToGlobalOp>(producer)) {
      return false;
    }
    if (isa<DistributedAwait, DistributedKernelOp>(producer)) {
      return true;
    }
    if (auto anchor = dyn_cast<AnchorPartitioningOp>(producer)) {
      value = anchor.getInput();
      continue;
    }
    return false;
  }
  return false;
}

static IndexedTensorShardingAttr buildDefaultShardingForType(MLIRContext *ctx,
                                                             Type type) {
  auto emptyAxes = DenseI64ArrayAttr::get(ctx, llvm::ArrayRef<int64_t>{});
  SmallVector<DenseI64ArrayAttr> dimPartitioningAxes;
  if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
    dimPartitioningAxes.reserve(rankedType.getRank());
    for (int64_t i = 0; i < rankedType.getRank(); ++i) {
      dimPartitioningAxes.push_back(emptyAxes);
    }
  }
  return IndexedTensorShardingAttr::get(ctx, dimPartitioningAxes, emptyAxes);
}

// A group of ops that will run as one kernel, tracked while clustering.
struct ColorState {
  int64_t id;
  llvm::SmallVector<Operation *> members;
  llvm::DenseSet<Operation *> externalProducers;
  llvm::DenseSet<Operation *> externalConsumers;
};

// Whether walking from `start` along operands (`upstream`) or along uses
// reaches an op of color `colorId`, through ops of `block`. Every finished
// color in `colors` is walked as one node, since its members run in one kernel:
// reaching any member reaches all of them.
//
// Worst case this visits every op in the block, so it is O(N) in the number of
// ops.
static bool reachesColor(Operation *start, int64_t colorId, bool upstream,
                         Block *block,
                         const llvm::DenseMap<Operation *, int64_t> &opToColor,
                         llvm::ArrayRef<ColorState> colors) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  llvm::SmallVector<Operation *> worklist{start};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second) {
      continue;
    }
    if (auto it = opToColor.find(op); it != opToColor.end()) {
      if (it->second == colorId) {
        return true;
      }
      if (it->second < static_cast<int64_t>(colors.size())) {
        worklist.append(colors[it->second].members.begin(),
                        colors[it->second].members.end());
      }
    }
    if (upstream) {
      for (Value operand : op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (producer && producer->getBlock() == block) {
          worklist.push_back(producer);
        }
      }
    } else {
      for (Operation *user : op->getUsers()) {
        if (user->getBlock() == block) {
          worklist.push_back(user);
        }
      }
    }
  }
  return false;
}

// Whether adding `candidate` to `color` would make two kernels depend on each
// other, counting every color finished so far as a single node.
//
// The op-level partial order alone cannot rule this out. Two ops with no
// dependence between them can each be linked to a third color: A -> X in one
// kernel and Y -> B in another are both fine op by op, but if A and B share a
// kernel, and X and Y share another, the kernels feed each other. So a
// candidate is rejected when one of its producers outside the color already
// depends on a member, or one of its consumers outside the color feeds one.
//
// Complexity: O(N^2) in the number of ops N in `block`. Each check walks up to
// the whole block once per operand and user of the candidate, and one check
// runs per candidate per sweep. This pass runs once per compilation, so this
// is not optimized; an incremental reachability structure over the contracted
// graph would make it near-linear.
static bool wouldCloseKernelCycle(
    Operation *candidate, const ColorState &color, Block *block,
    const llvm::DenseMap<Operation *, int64_t> &opToColor,
    llvm::ArrayRef<ColorState> colors) {
  auto isExternal = [&](Operation *op) {
    auto it = opToColor.find(op);
    return it == opToColor.end() || it->second != color.id;
  };
  for (Value operand : candidate->getOperands()) {
    Operation *producer = operand.getDefiningOp();
    if (producer && producer->getBlock() == block && isExternal(producer) &&
        reachesColor(producer, color.id, /*upstream=*/true, block, opToColor,
                     colors)) {
      return true;
    }
  }
  for (Operation *user : candidate->getUsers()) {
    if (user->getBlock() == block && isExternal(user) &&
        reachesColor(user, color.id, /*upstream=*/false, block, opToColor,
                     colors)) {
      return true;
    }
  }
  return false;
}

struct ClusterDistributedKernelsPass
    : public impl::ClusterDistributedKernelsPassBase<
          ClusterDistributedKernelsPass> {
  using ClusterDistributedKernelsPassBase::ClusterDistributedKernelsPassBase;
  using TV_AxisFactor = mlir::TypedValue<mlir::enzyme::axis::AxisFactorType>;
  using TV_FactorGroup = mlir::TypedValue<mlir::enzyme::axis::FactorGroupType>;

  llvm::DenseMap<AxisSymbol, TV_AxisFactor> symbolToLogicalAxis;
  ShardyLogicalAxisAnalysis axisAnalysis;

  // optional just so we can have a default constructor. We will set this in
  // runOnOperation.
  std::optional<OpBuilder> axis_builder;
  std::optional<mlir::Location> axis_loc;

  ClusterDistributedKernelsPass() = default;

  // Materializes an axis value for a symbol not already anchored in the IR: a
  // LogicalMeshAxesOp, or a DeviceLocalAxisOp for a symbol that must never be
  // sharded across the mesh.
  TV_AxisFactor getOrCreateLogicalAxisForSymbol(AxisSymbol symbol) {
    // if present, return
    auto it = symbolToLogicalAxis.find(symbol);
    if (it != symbolToLogicalAxis.end()) {
      return it->second;
    }
    if (Value logicalAxis = axisAnalysis.getLogicalAxis(symbol)) {
      auto factor = cast<TV_AxisFactor>(logicalAxis);
      symbolToLogicalAxis[symbol] = factor;
      return factor;
    }
    // otherwise, we need to instantiate a new axis,
    // then turn it into a factor.
    Value axisValue;
    if (axisAnalysis.isUnshardable(symbol)) {
      axisValue =
          axis_builder
              ->create<mlir::enzyme::distributed::DeviceLocalAxisOp>(
                  *axis_loc, symbol.getExtent())
              .getAxis();
    } else {
      axisValue =
          axis_builder
              ->create<mlir::enzyme::distributed::LogicalMeshAxesOp>(
                  *axis_loc, symbol.getExtent())
              .getAxis();
    }
    auto as_factor =
        axis::viewAxisAsFactor(axisValue, *axis_builder, *axis_loc);
    symbolToLogicalAxis[symbol] = as_factor;
    return as_factor;
  }

  llvm::SmallVector<TV_AxisFactor>
  getLogicalAxesForSymbols(llvm::ArrayRef<AxisSymbol> symbols) {
    llvm::SmallVector<TV_AxisFactor> logical_axes;
    logical_axes.reserve(symbols.size());
    for (const auto &symbol : symbols) {
      logical_axes.push_back(getOrCreateLogicalAxisForSymbol(symbol));
    }
    return logical_axes;
  }

  // Normalizes one logical symbol into a single-factor group for
  // distributed.function partitioning_axes metadata.
  TV_FactorGroup getOrCreatePartitioningAxisGroup(AxisSymbol symbol) {
    auto factor = getOrCreateLogicalAxisForSymbol(symbol);
    return axis::viewFactorsAsProduct(factor, *axis_builder, *axis_loc);
  }

  // Builds one positional factor group per tensor dimension for a real
  // global/local Cast op. Each group preserves the logical-axis provenance
  // of that dimension (mirrors MaterializeDistributedCollectives.cpp's
  // helper of the same name).
  SmallVector<Value> getTensorPartitioningAxisGroups(
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    SmallVector<Value> partitioningAxisGroups;
    partitioningAxisGroups.reserve(partitioningAxes.size());
    for (const auto &dimensionAxes : partitioningAxes) {
      auto factors = getLogicalAxesForSymbols(dimensionAxes);
      partitioningAxisGroups.push_back(
          axis::viewFactorsAsProduct(factors, *axis_builder, *axis_loc));
    }
    return partitioningAxisGroups;
  }

  IndexedTensorShardingAttr buildIndexedShardingAttr(
      RankedTensorType tensorType,
      const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
          &partitioningDims,
      llvm::DenseMap<AxisSymbol, int64_t> &symbolToPartitioningAxisIdx) {
    auto *ctx = &getContext();
    SmallVector<DenseI64ArrayAttr> dimPartitioningAxes;
    dimPartitioningAxes.reserve(tensorType.getRank());

    for (int64_t dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
      SmallVector<int64_t> partitioningAxisIndices;
      if (dimIdx < static_cast<int64_t>(partitioningDims.size())) {
        for (AxisSymbol symbol : partitioningDims[dimIdx]) {
          auto [it, inserted] = symbolToPartitioningAxisIdx.try_emplace(
              symbol, symbolToPartitioningAxisIdx.size());
          (void)inserted;
          partitioningAxisIndices.push_back(it->second);
        }
      }
      dimPartitioningAxes.push_back(
          DenseI64ArrayAttr::get(ctx, partitioningAxisIndices));
    }

    return IndexedTensorShardingAttr::get(
        ctx, dimPartitioningAxes,
        DenseI64ArrayAttr::get(ctx, llvm::ArrayRef<int64_t>{}));
  }

  LogicalResult clusterOpsIntoKernels(Block *mainBlock,
                                      PartialOrder<Operation *> &order,
                                      ShardyLogicalAxisAnalysis &axisAnalysis) {
    // Partition compute operations into groups that can be outlined into
    // distributed.kernel regions without changing dataflow ordering.
    //
    // Preconditions:
    // - Layout/reduction collectives have already been materialized.
    // - IR may still contain unrealized conversion casts between global and
    //   local tensor views.
    //
    // Clustering model:
    // - Only clusterable compute ops participate. Communication/control/meta
    //   ops stay outside kernels and act as boundaries.
    // - Candidate ops are first bucketed by partitioning-axis basis
    //   (flattened, sorted unique AxisSymbol set).
    // - Each bucket is processed in block order. A color starts from the first
    //   uncolored op, then repeatedly sweeps remaining bucket members.
    // - A candidate is accepted if it does not violate topology boundaries of
    //   the color: it must not be before any external producer boundary and
    //   must not be after any external consumer boundary. It must also not
    //   make two kernels depend on each other (see wouldCloseKernelCycle).
    // - After each accepted op, producer/consumer boundaries are recomputed
    //   from direct SSA use/def edges. Sweeps continue until a fixed point
    //   is reached.
    //
    // After colors are finalized, one distributed.kernel is materialized per
    // color and the block is topologically reordered.
    using PartitioningAxes =
        ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis;

    struct BucketState {
      llvm::SmallVector<Operation *> pending;
    };

    auto isClusterableOp = [](Operation *op) {
      // Keep communication and control/meta ops outside kernels.
      if (isa<DistributedCollectiveOp, DistributedAwait, DistributedYieldOp,
              DistributedKernelOp, UnrealizedConversionCastOp,
              DistributedCastGlobalToLocalOp, DistributedCastLocalToGlobalOp,
              AnchorPartitioningOp>(op)) {
        return false;
      }

      // Axis dialect ops are sharding metadata plumbing, not computation.
      if (op->getName().getDialectNamespace() == "axis") {
        return false;
      }

      return true;
    };

    // build the string key for the map
    auto canonicalBasisKey = [&](const PartitioningAxes &axes) {
      llvm::SmallVector<std::pair<uint64_t, uint64_t>> symbols;
      for (const auto &axis : axes) {
        for (const AxisSymbol &symbol : axis) {
          // Unshardable axes are device-local, so never need communication:
          // ops can share a kernel whatever their unshardable axes are.
          if (!axisAnalysis.isUnshardable(symbol)) {
            symbols.emplace_back(symbol.getId(), symbol.getExtent());
          }
        }
      }
      std::sort(symbols.begin(), symbols.end());
      symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());

      std::string key;
      llvm::raw_string_ostream os(key);
      for (auto [id, extent] : symbols) {
        os << id << ":" << extent << "|";
      }
      os.flush();
      return key;
    };

    auto tryAcceptCandidate =
        [&](Operation *candidate, ColorState &color,
            llvm::DenseMap<Operation *, int64_t> &opToColor,
            llvm::ArrayRef<ColorState> colors) {
          if (!candidate || !isClusterableOp(candidate) ||
              opToColor.contains(candidate)) {
            return false;
          }

          // The boundary checks below only compare against boundary ops
          // recorded for the members so far, not the candidate's own, and
          // cannot see cycles through other kernels.
          if (wouldCloseKernelCycle(candidate, color, mainBlock, opToColor,
                                    colors)) {
            return false;
          }

          // Respect color boundary topology constraints.
          for (Operation *externalProducer : color.externalProducers) {
            if (order.compare(candidate, externalProducer) == Order::LessThan) {
              return false;
            }
          }
          for (Operation *externalConsumer : color.externalConsumers) {
            if (order.compare(candidate, externalConsumer) ==
                Order::GreaterThan) {
              return false;
            }
          }

          opToColor[candidate] = color.id;
          color.members.push_back(candidate);
          color.externalProducers.erase(candidate);
          color.externalConsumers.erase(candidate);
          return true;
        };

    auto updateBoundary = [&](Operation *accepted, ColorState &color,
                              llvm::DenseMap<Operation *, int64_t> &opToColor) {
      for (OpOperand &operand : accepted->getOpOperands()) {
        Operation *producer = operand.get().getDefiningOp();
        if (!producer || producer->getBlock() != mainBlock) {
          continue;
        }
        auto producerColor = opToColor.find(producer);
        if (producerColor == opToColor.end() ||
            producerColor->second != color.id) {
          color.externalProducers.insert(producer);
        }
      }

      for (Value result : accepted->getResults()) {
        for (OpOperand &use : result.getUses()) {
          Operation *consumer = use.getOwner();
          if (!consumer || consumer->getBlock() != mainBlock) {
            continue;
          }
          auto consumerColor = opToColor.find(consumer);
          if (consumerColor == opToColor.end() ||
              consumerColor->second != color.id) {
            color.externalConsumers.insert(consumer);
          }
        }
      }
    };

    llvm::DenseMap<Operation *, int64_t> opToColor;
    llvm::SmallVector<ColorState> colors;
    // Use a canonical string key to keep bucket lookup simple with
    // llvm::StringMap and avoid custom DenseMap hashing for vector keys.
    llvm::StringMap<int64_t> keyToBucket;
    llvm::SmallVector<BucketState> buckets;
    int64_t nextColorId = 0;

    // Phase 1: collect clusterable ops into basis buckets in block order.
    // Only valid, clusterable ops are inserted, so sweep logic can operate
    // directly on the remaining bucket list without extra candidate cleanup.
    for (Operation &opRef : mainBlock->getOperations()) {
      Operation *op = &opRef;
      if (!isClusterableOp(op)) {
        continue;
      }
      PartitioningAxes axes = axisAnalysis.getPartitioningAxes(op);
      std::string key = canonicalBasisKey(axes);

      auto it = keyToBucket.find(key);
      if (it == keyToBucket.end()) {
        int64_t bucketIdx = static_cast<int64_t>(buckets.size());
        keyToBucket[key] = bucketIdx;
        buckets.push_back(BucketState{{}});
        it = keyToBucket.find(key);
      }
      buckets[it->second].pending.push_back(op);
    }

    // Phase 2: form colors with fixed-point sweeps inside each bucket.
    for (BucketState &bucket : buckets) {

      while (!bucket.pending.empty()) {
        ColorState color{nextColorId++, {}, {}, {}};
        size_t pendingSizeBeforeColor = bucket.pending.size();

        // Repeatedly sweep remaining bucket members until no further
        // candidates can be accepted into this color.
        bool changed = true;
        while (changed) {
          changed = false;

          for (size_t i = 0; i < bucket.pending.size();) {
            Operation *candidate = bucket.pending[i];
            if (!tryAcceptCandidate(candidate, color, opToColor, colors)) {
              ++i;
              continue;
            }

            updateBoundary(candidate, color, opToColor);
            bucket.pending.erase(bucket.pending.begin() + i);
            changed = true;
          }
        }

        if (bucket.pending.size() >= pendingSizeBeforeColor) {
          mainBlock->getParentOp()->emitError()
              << "bucket sweep made no progress while pending ops remain";
          return failure();
        }
        assert(!color.members.empty() &&
               "color formation must accept at least one pending op");
        assert(bucket.pending.size() < pendingSizeBeforeColor &&
               "bucket pending list must shrink after each color");

        colors.push_back(std::move(color));
      }
    }

    // Phase 3: assert every clusterable op was assigned to some color.
    for (Operation &opRef : mainBlock->getOperations()) {
      Operation *op = &opRef;
      if (!isClusterableOp(op)) {
        continue;
      }
      if (!opToColor.contains(op)) {
        op->emitError() << "clusterable op was not assigned a kernel color";
        return failure();
      }
    }

    auto *ctx = &getContext();

    // NOTE: Kernel boundary values that are ranked tensors are expected to
    // carry sharding information. We recover this from direct analysis,
    // representative use-sites, and nearby unrealized casts; if still
    // missing, treat it as an invariant violation for compute values. We only
    // allow default-empty sharding for non-ranked values.
    // Phase 4: materialize one distributed.kernel for each color.
    OpBuilder builder(ctx);
    for (size_t colorIdx = 0; colorIdx < colors.size(); ++colorIdx) {
      const ColorState &color = colors[colorIdx];
      if (color.members.empty()) {
        continue;
      }

      llvm::DenseSet<Operation *> memberSet(color.members.begin(),
                                            color.members.end());

      llvm::SmallVector<Operation *> orderedMembers;
      orderedMembers.reserve(color.members.size());
      for (Operation &op : mainBlock->getOperations()) {
        if (memberSet.contains(&op)) {
          orderedMembers.push_back(&op);
        }
      }
      if (orderedMembers.empty()) {
        continue;
      }

      llvm::DenseMap<Value, int64_t> inputIndices;
      llvm::SmallVector<Value> kernelInputs;
      // with casts and collectives inserted the producer may not be known to
      // the logical axis analysis for a value, but the use should be,
      // allowing us to look up partitioning axes.
      llvm::DenseMap<Value, OpOperand *> representativeInputUse;
      llvm::DenseMap<Value, int64_t> outputIndices;
      llvm::SmallVector<Value> kernelOutputs;

      for (Operation *member : orderedMembers) {
        for (OpOperand &operand : member->getOpOperands()) {
          Value input = operand.get();
          Operation *producer = input.getDefiningOp();
          bool producedInsideColor = producer &&
                                     producer->getBlock() == mainBlock &&
                                     memberSet.contains(producer);
          if (producedInsideColor) {
            continue;
          }
          if (!inputIndices.contains(input)) {
            inputIndices[input] = static_cast<int64_t>(kernelInputs.size());
            kernelInputs.push_back(input);
            representativeInputUse[input] = &operand;
          }
        }
      }

      for (Operation *member : orderedMembers) {
        for (Value result : member->getResults()) {
          bool escapesColor = false;
          for (OpOperand &use : result.getUses()) {
            Operation *consumer = use.getOwner();
            bool consumedInsideColor = consumer &&
                                       consumer->getBlock() == mainBlock &&
                                       memberSet.contains(consumer);
            if (!consumedInsideColor) {
              escapesColor = true;
              break;
            }
          }
          if (!escapesColor) {
            continue;
          }
          if (!outputIndices.contains(result)) {
            outputIndices[result] = static_cast<int64_t>(kernelOutputs.size());
            kernelOutputs.push_back(result);
          }
        }
      }

      SmallVector<Value> kernelInputOperands;
      kernelInputOperands.reserve(kernelInputs.size());
      SmallVector<Type> kernelBlockArgTypes;
      kernelBlockArgTypes.reserve(kernelInputs.size());

      llvm::DenseMap<AxisSymbol, int64_t> symbolToPartitioningAxisIdx;
      SmallVector<IndexedTensorShardingAttr> inputShardings;
      inputShardings.reserve(kernelInputs.size());

      Operation *insertBefore = mainBlock->getTerminator();
      builder.setInsertionPoint(insertBefore);

      for (Value input : kernelInputs) {
        kernelBlockArgTypes.push_back(input.getType());

        // Prefer `input`'s own producer-tracked identity over the consuming
        // op's synthesized per-factor symbol: the latter always resolves
        // structurally for any op with a sharding rule, even when nothing
        // ever unioned it back to the value's true identity, which would
        // silently materialize a phantom partitioning axis if tried first.
        auto maybePartitioning =
            getPartitioningForValueOrCastNeighborhood(input, axisAnalysis);
        if (!maybePartitioning) {
          maybePartitioning =
              axisAnalysis.getTensorPartitionDims(*representativeInputUse[input]);
        }

        if (!maybePartitioning && isa<RankedTensorType>(input.getType())) {
          insertBefore->emitError()
              << "missing sharding for ranked kernel input value " << input;
          return failure();
        }

        // `input` itself may already be local-scoped (e.g. it flows straight
        // from a collective's Await, possibly through a scope-agnostic
        // AnchorPartitioning) -- see isLocallyScopedValue's comment. In that
        // case its own type already IS the local type; dividing it by the
        // partitioning extent again would be wrong (and, since a fully-local
        // value's size is already <= the divided target, silently produces
        // degenerate near-zero dimensions). Only genuinely global values get
        // divided down.
        Type localInputType = input.getType();
        if (maybePartitioning && !isLocallyScopedValue(input)) {
          localInputType = getLocalTypeForValue(input, *maybePartitioning);
        }

        Value localInputValue = input;
        if (localInputValue.getType() != localInputType) {
          SmallVector<Value> axesOperands =
              getTensorPartitioningAxisGroups(*maybePartitioning);
          localInputValue = builder
                                .create<DistributedCastGlobalToLocalOp>(
                                    insertBefore->getLoc(), localInputType,
                                    localInputValue, axesOperands)
                                .getOutput();
        }
        kernelInputOperands.push_back(localInputValue);

        if (auto rankedType = dyn_cast<RankedTensorType>(localInputType);
            rankedType && maybePartitioning) {
          inputShardings.push_back(buildIndexedShardingAttr(
              rankedType, *maybePartitioning, symbolToPartitioningAxisIdx));
        } else {
          inputShardings.push_back(
              buildDefaultShardingForType(ctx, localInputType));
        }
      }

      SmallVector<IndexedTensorShardingAttr> outputShardings;
      outputShardings.reserve(kernelOutputs.size());
      SmallVector<Type> kernelResultTypes;
      kernelResultTypes.reserve(kernelOutputs.size());
      for (Value output : kernelOutputs) {
        auto maybePartitioning =
            getPartitioningForValueOrCastNeighborhood(output, axisAnalysis);
        if (!maybePartitioning && isa<RankedTensorType>(output.getType())) {
          insertBefore->emitError()
              << "missing sharding for ranked kernel output value " << output;
          return failure();
        }
        // Kernel returns are in local type, global type recoverable
        // from yield or from multiplying the local type by the sharding.
        Type globalOutputType = output.getType();
        Type localOutputType = getLocalTypeForValue(output, *maybePartitioning);
        kernelResultTypes.push_back(localOutputType);

        if (auto rankedType = dyn_cast<RankedTensorType>(globalOutputType);
            rankedType && maybePartitioning) {
          outputShardings.push_back(buildIndexedShardingAttr(
              rankedType, *maybePartitioning, symbolToPartitioningAxisIdx));
        } else {
          outputShardings.push_back(
              buildDefaultShardingForType(ctx, globalOutputType));
        }
      }

      // Every AxisSymbol referenced anywhere in this kernel -- by its own
      // boundary values above or by any member's operand/result below --
      // must be registered in symbolToPartitioningAxisIdx before
      // kernelPartitioningAxes is sized from that map. buildIndexedShardingAttr
      // assigns a symbol its index the first time it's seen, so a symbol that
      // only appears inside the body (e.g. an extent-1 factor a broadcast's
      // operand carries but the kernel's boundary values don't) needs to be
      // registered here; otherwise its dim_partitioning_axes index would point
      // past the kernel's own `axes` operand list, with nothing downstream
      // able to resolve it. The attrs computed here are cached and reused
      // verbatim once members are cloned below, so this pass isn't redundant
      // work.
      llvm::DenseMap<Operation *,
                     std::pair<SmallVector<IndexedTensorShardingAttr>,
                               SmallVector<IndexedTensorShardingAttr>>>
          memberShardings;
      for (Operation *member : orderedMembers) {
        SmallVector<IndexedTensorShardingAttr> argumentShardings;
        argumentShardings.reserve(member->getNumOperands());
        for (OpOperand &operand : member->getOpOperands()) {
          auto maybePartitioning = axisAnalysis.getTensorPartitionDims(operand);
          Type operandType = operand.get().getType();
          if (!maybePartitioning && isa<RankedTensorType>(operandType)) {
            member->emitError()
                << "missing sharding for ranked kernel operation operand "
                << operand.getOperandNumber();
            return failure();
          }

          if (auto rankedType = dyn_cast<RankedTensorType>(operandType);
              rankedType && maybePartitioning) {
            argumentShardings.push_back(buildIndexedShardingAttr(
                rankedType, *maybePartitioning, symbolToPartitioningAxisIdx));
          } else {
            argumentShardings.push_back(
                buildDefaultShardingForType(ctx, operandType));
          }
        }

        SmallVector<IndexedTensorShardingAttr> outputShardings;
        outputShardings.reserve(member->getNumResults());
        for (OpResult result : member->getResults()) {
          auto maybePartitioning = axisAnalysis.getTensorPartitionDims(result);
          Type resultType = result.getType();
          if (!maybePartitioning && isa<RankedTensorType>(resultType)) {
            member->emitError()
                << "missing sharding for ranked kernel operation result "
                << result.getResultNumber();
            return failure();
          }

          if (auto rankedType = dyn_cast<RankedTensorType>(resultType);
              rankedType && maybePartitioning) {
            outputShardings.push_back(buildIndexedShardingAttr(
                rankedType, *maybePartitioning, symbolToPartitioningAxisIdx));
          } else {
            outputShardings.push_back(
                buildDefaultShardingForType(ctx, resultType));
          }
        }

        memberShardings[member] = {std::move(argumentShardings),
                                   std::move(outputShardings)};
      }

      SmallVector<Value> kernelPartitioningAxes(
          symbolToPartitioningAxisIdx.size());
      for (const auto &[symbol, idx] : symbolToPartitioningAxisIdx) {
        kernelPartitioningAxes[idx] = getOrCreatePartitioningAxisGroup(symbol);
      }

      auto inputShardingsAttr =
          IndexedTensorShardingPerValueAttr::get(ctx, inputShardings);
      auto outputShardingsAttr =
          IndexedTensorShardingPerValueAttr::get(ctx, outputShardings);

      auto kernel = builder.create<DistributedKernelOp>(
          insertBefore->getLoc(), TypeRange(kernelResultTypes),
          ValueRange(kernelInputOperands), ValueRange(kernelPartitioningAxes),
          inputShardingsAttr, outputShardingsAttr);

      Region &kernelBody = kernel->getRegion(0);
      if (kernelBody.empty()) {
        kernelBody.push_back(new Block());
      }
      Block &kernelBlock = kernelBody.front();
      for (Type argType : kernelBlockArgTypes) {
        kernelBlock.addArgument(argType, insertBefore->getLoc());
      }

      IRMapping mapping;
      for (auto [idx, input] : llvm::enumerate(kernelInputs)) {
        mapping.map(input, kernelBlock.getArgument(idx));
      }

      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&kernelBlock);
      for (Operation *member : orderedMembers) {
        Operation *cloned = bodyBuilder.clone(*member, mapping);

        // Shardings were already computed (and symbolToPartitioningAxisIdx
        // already fully populated) by the pre-scan above; annotate the
        // clone with them using the kernel-wide partitioning-axis index
        // space, for looking up the partitioning of internal values in the
        // kernel body later (i.e. for doing the actual sharding rewrite).
        const auto &[argumentShardings, outputShardings] =
            memberShardings.at(member);
        cloned->setAttr(
            "distributed.argument_shardings",
            IndexedTensorShardingPerValueAttr::get(ctx, argumentShardings));
        cloned->setAttr(
            "distributed.output_shardings",
            IndexedTensorShardingPerValueAttr::get(ctx, outputShardings));
      }

      SmallVector<Value> yieldedValues;
      yieldedValues.reserve(kernelOutputs.size());
      for (Value output : kernelOutputs) {
        yieldedValues.push_back(mapping.lookup(output));
      }
      bodyBuilder.setInsertionPointToEnd(&kernelBlock);
      bodyBuilder.create<DistributedYieldOp>(insertBefore->getLoc(),
                                             yieldedValues);

      for (auto [resultIdx, oldValue] : llvm::enumerate(kernelOutputs)) {
        Value newValue = kernel.getResult(resultIdx);
        for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
          Operation *consumer = use.getOwner();
          bool consumedInsideColor = consumer &&
                                     consumer->getBlock() == mainBlock &&
                                     memberSet.contains(consumer);
          if (!consumedInsideColor) {
            Type expectedUseType = use.get().getType();
            Value valueForUse = newValue;
            if (valueForUse.getType() != expectedUseType) {
              builder.setInsertionPoint(use.getOwner());
              // The kernel's own result is local; the outside use still
              // expects the pre-clustering global type. oldValue's defining
              // op hasn't been erased yet (that happens below), so its
              // partitioning is still resolvable here.
              auto maybePartitioning =
                  getPartitioningForValueOrCastNeighborhood(oldValue,
                                                            axisAnalysis);
              if (!maybePartitioning) {
                insertBefore->emitError()
                    << "missing sharding to cast kernel result back to its "
                       "pre-clustering type "
                    << oldValue;
                return failure();
              }
              SmallVector<Value> axesOperands =
                  getTensorPartitioningAxisGroups(*maybePartitioning);
              valueForUse = builder
                                .create<DistributedCastLocalToGlobalOp>(
                                    insertBefore->getLoc(), expectedUseType,
                                    valueForUse, axesOperands)
                                .getOutput();
            }
            use.set(valueForUse);
          }
        }
      }

      for (auto it = orderedMembers.rbegin(); it != orderedMembers.rend();
           ++it) {
        (*it)->erase();
      }
    }

    // Phase 5: normalize final block order after outlining.
    if (!mlir::sortTopologically(mainBlock)) {
      mainBlock->getParentOp()->emitError()
          << "failed to reorder block operations topologically";
      return failure();
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();

    const auto &mainFunctionAnalysis = getAnalysis<FindMainFunctionAnalysis>();
    if (!mainFunctionAnalysis.isValid()) {
      if (!mainFunctionAnalysis.hasMainFunction()) {
        emitWarning(module_op.getLoc())
            << "no main function found; skipping pass";
        return;
      }
      emitError(module_op.getLoc())
          << "multiple symbols named 'main' found across func.func and "
             "distributed.DistributedFunction";
      signalPassFailure();
      return;
    }

    Operation *mainScopeOp = mainFunctionAnalysis.getMainFunctionOp();
    if (!mainScopeOp) {
      emitWarning(module_op.getLoc())
          << "main is not a function-like op; skipping pass";
      return;
    }

    Block *mainBlock = mainFunctionAnalysis.getMainBlock();
    if (!mainBlock) {
      mainScopeOp->emitRemark()
          << "main is not a func.func with a single-block body; skipping pass";
      signalPassFailure();
      return;
    }

    // prep for building axes
    axis_builder = OpBuilder(module_op.getContext());
    axis_builder->setInsertionPointToStart(&module_op.getBodyRegion().front());
    axis_loc = mainScopeOp->getLoc();

    const auto &mainAxisAnalysis =
        getAnalysis<MainFunctionShardyLogicalAxisAnalysis>();
    if (!mainAxisAnalysis.isValid()) {
      mainScopeOp->emitRemark()
          << "failed to build module-scoped main logical axis analysis";
      signalPassFailure();
      return;
    }
    axisAnalysis = mainAxisAnalysis.getAnalysis();

    if (dumpLogicalAxes) {
      dumpLogicalAxesForMainBlock(mainBlock, axisAnalysis);
    }

    // need partial order to help clustering: need to ensure that nothing
    // outside a kernel sits topologically between any part of a kernel.
    auto &order_analysis =
        getAnalysis<MainFunctionSSABlockPartialOrderAnalysis>();
    if (!order_analysis.isValid()) {
      emitError(module_op.getLoc())
          << "failed to build module-scoped main SSA partial order analysis";
      signalPassFailure();
      return;
    }
    auto &order = order_analysis.getPartialOrder();
    if (failed(clusterOpsIntoKernels(mainBlock, order, axisAnalysis))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
