#include "src/enzyme_ad/jax/Passes/Distributed/LocalizationPolicy.h"
#include "src/enzyme_ad/jax/Passes/Distributed/PartitioningPolicy.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>
#include <memory>

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_LOCALIZEDISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct PlannedLocalization {
  OpRange range;
  llvm::SmallVector<TypedValue<LogicalCommAxisType>> localization;
  llvm::SmallVector<Operation *> ops;
};

struct SplitBlocks {
  Block *pre = nullptr;
  Block *range = nullptr;
  Block *post = nullptr;
};

llvm::SmallVector<Operation *> snapshotRangeOps(OpRange range) {
  llvm::SmallVector<Operation *> ops;
  for (OpIterator it = range.begin; it != range.end; ++it) {
    ops.push_back(&*it);
  }
  return ops;
}

/**
 * For an op range over a block, if the last op in the range is the block
 * terminator, return a new range that excludes the terminator.
 */
OpRange excludeTerminatorFromRange(OpRange range) {
  OpIterator adjustedEnd = range.end;
  if (range.begin == adjustedEnd) {
    return OpRange{range.begin, adjustedEnd};
  }

  Block *block = (&*range.begin)->getBlock();
  assert(block);

  if (adjustedEnd == block->end()) {
    OpIterator terminatorIt = Block::iterator(block->getTerminator());
    adjustedEnd = terminatorIt;
  }

  return OpRange{range.begin, adjustedEnd};
}

/**
 * Split a block around a given range. The newly created blocks will
 * not have terminators. If there are no operations before or after the range,
 * this function will still create new empty blocks. Fails for
 * an empty range.
 */
FailureOr<SplitBlocks>
splitPlannedRangeIntoBlocks(const PlannedLocalization &planned) {
  if (planned.ops.empty()) {
    return failure();
  }

  Operation *firstOp = planned.ops.front();
  Operation *lastOp = planned.ops.back();
  Block *originalBlock = firstOp->getBlock();
  assert(originalBlock);

  SplitBlocks split;
  split.pre = originalBlock;
  split.range = originalBlock->splitBlock(Block::iterator(firstOp));

  Operation *afterLast = lastOp->getNextNode();
  if (afterLast) {
    split.post = split.range->splitBlock(Block::iterator(afterLast));
  } else {
    split.post = split.range->splitBlock(split.range->end());
  }
  return split;
}

/**
 * Filter a list of values to only include those of tensor type.
 */
llvm::SmallVector<Value> filterTensorValues(ValueRange values) {
  llvm::SmallVector<Value> tensorValues;
  for (Value value : values) {
    if (isa<TensorType>(value.getType())) {
      tensorValues.push_back(value);
    }
  }
  return tensorValues;
}

/**
 * Find live out definitions in a block, as opposed to values that are live out
 * but defined outside the block.
 */
llvm::SmallVector<Value>
filterLiveOutsDefinedInRange(Block *rangeBlock, ArrayRef<Value> liveOuts) {
  llvm::SmallPtrSet<Operation *, 16> rangeOps;
  for (Operation &op : *rangeBlock) {
    rangeOps.insert(&op);
  }

  llvm::SmallVector<Value> filtered;
  for (Value value : liveOuts) {
    Operation *defOp = value.getDefiningOp();
    if (defOp && rangeOps.contains(defOp)) {
      filtered.push_back(value);
    }
  }
  return filtered;
}

/**
 * Maps the localized range into each region. Replaces the live in tensor values
 * with the corresponding region arguments, leaving live-in axis and mesh values
 * as-is. TODO: at this stage we can also consider changing the tensor types
 * from their global view to their local view.
 */
void cloneRangeIntoRegionComputationBodies(
    RegionComputationOp regionComputation, Block *rangeBlock,
    ArrayRef<Value> liveIns, ArrayRef<Value> liveOuts,
    const llvm::DenseMap<Value, llvm::SmallVector<Value>> &deviceTokenMap,
    Location loc) {
  llvm::SmallVector<Type> liveInTypes;
  liveInTypes.reserve(liveIns.size());
  for (Value value : liveIns) {
    liveInTypes.push_back(value.getType());
  }
  llvm::SmallVector<Location> liveInLocs(liveIns.size(), loc);

  for (auto [deviceIndex, deviceBody] :
       llvm::enumerate(regionComputation.getDeviceBodies())) {
    auto *deviceBlock = new Block();
    deviceBlock->addArguments(liveInTypes, liveInLocs);
    deviceBody.push_back(deviceBlock);

    IRMapping mapper;
    for (auto [index, liveIn] : llvm::enumerate(liveIns)) {
      mapper.map(liveIn, deviceBlock->getArgument(index));
    }

    for (const auto &entry : deviceTokenMap) {
      Value collectiveToken = entry.first;
      if (deviceIndex >= entry.second.size()) {
        continue;
      }
      mapper.map(collectiveToken, entry.second[deviceIndex]);
    }

    for (Operation &op : *rangeBlock) {
      if (op.hasTrait<mlir::OpTrait::enzyme::distributed::MetadataTrait>()) {
        continue;
      }
      deviceBlock->push_back(op.clone(mapper));
    }

    llvm::SmallVector<Value> yieldedValues;
    yieldedValues.reserve(liveOuts.size());
    for (Value liveOut : liveOuts) {
      yieldedValues.push_back(mapper.lookupOrDefault(liveOut));
    }

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(deviceBlock);
    bodyBuilder.create<DistributedYieldOp>(loc, yieldedValues);
  }
}

/**
 * When mapping into local regions, builds the datastructure
 * to translate global collective tokens to device-specific parts.
 */
FailureOr<llvm::DenseMap<Value, llvm::SmallVector<Value>>>
buildDeviceTokenMap(ArrayRef<CollectiveOp> collectiveOps, Value submesh,
                    unsigned nDevices, OpBuilder &builder, Location loc) {
  llvm::DenseMap<Value, llvm::SmallVector<Value>> deviceTokenMap;
  auto ip = builder.saveInsertionPoint();

  for (CollectiveOp collective : collectiveOps) {
    builder.setInsertionPointAfter(collective);
    auto tokenParts = builder.create<SubmeshCollectivePartsOp>(
        loc, collective.getToken(), submesh,
        collective.getLocalInputTensorTypeAttr(),
        collective.getLocalOutputTensorTypeAttr());

    llvm::SmallVector<Value> deviceTokens(
        tokenParts.getDeviceSpecificTokens().begin(),
        tokenParts.getDeviceSpecificTokens().end());
    if (deviceTokens.size() != nDevices) {
      tokenParts.emitError()
          << "unexpected number of device-specific tokens for submesh";
      return failure();
    }
    deviceTokenMap[collective.getToken()] = std::move(deviceTokens);
  }
  builder.restoreInsertionPoint(ip);

  return deviceTokenMap;
}

/**
 * For a range being localized, metadata like axis definitions should be lifted
 * outside the range, allowing it to be visible for all regions.
 */
llvm::SmallVector<CollectiveOp> hoistMetadataOps(Block *fromBlock,
                                                 Block *toBlock) {
  llvm::SmallVector<CollectiveOp> hoistedCollectives;
  for (auto it = fromBlock->begin(), e = fromBlock->end(); it != e;) {
    Operation &op = *it++;
    if (!op.hasTrait<mlir::OpTrait::enzyme::distributed::MetadataTrait>()) {
      continue;
    }
    if (auto collective = dyn_cast<CollectiveOp>(op)) {
      hoistedCollectives.push_back(collective);
    }
    op.moveBefore(toBlock, toBlock->end());
  }
  return hoistedCollectives;
}

void replaceRangeWithRegionComputation(const SplitBlocks &split,
                                       RegionComputationOp regionComputation,
                                       ArrayRef<Value> liveOuts) {
  auto regionResults = regionComputation.getOutputTensors();
  if (regionResults.size() != liveOuts.size()) {
    return;
  }

  for (auto [index, liveOut] : llvm::enumerate(liveOuts)) {
    Value oldValue = liveOut;
    oldValue.replaceAllUsesWith(regionResults[index]);
  }

  Region *parentRegion = split.pre->getParent();
  if (!parentRegion) {
    return;
  }

  if (split.post) {
    split.pre->getOperations().splice(split.pre->end(),
                                      split.post->getOperations());
    parentRegion->getBlocks().erase(split.post->getIterator());
  }
  parentRegion->getBlocks().erase(split.range->getIterator());
}

} // namespace

LogicalResult localizeMeshComputation(MeshComputationOp meshComputationOp) {
  auto localizationPolicy =
      std::make_unique<NaiveLocalMostPolicy>(/*budget=*/2);
  auto partitioningPolicy =
      std::make_unique<TrivialWholeOpPartitioningPolicy>();

  llvm::SmallVector<OpRange> specializationRanges =
      partitioningPolicy->suggestSpecializationRanges(meshComputationOp);
  llvm::SmallVector<PlannedLocalization, 1> plannedLocalizations;
  plannedLocalizations.reserve(specializationRanges.size());

  for (auto &range : specializationRanges) {
    OpRange normalizedRange = excludeTerminatorFromRange(range);
    if (normalizedRange.begin == normalizedRange.end) {
      continue;
    }

    auto localization = localizationPolicy->suggestLocalization(
        meshComputationOp, normalizedRange.begin, normalizedRange.end);
    PlannedLocalization planned{normalizedRange, std::move(localization)};
    planned.ops = snapshotRangeOps(normalizedRange);
    if (planned.ops.empty()) {
      continue;
    }
    plannedLocalizations.push_back(std::move(planned));

    // Here we need to:
    // - Split the basic block at this range
    // - gather the live in and live out values for this range/basic block
    //   (there is an API for this)): they become the inputs and outputs for
    //   a new mesh computation region.
    // - Create a new logical mesh above the split basic block using the
    // suggested
    //   localization. Should be subseteq to the mesh computation mesh, which
    //   we can allow the verifiers to enforce but may rely on here. Record
    //   the size of the new mesh.
    // - Move any distributed.collective token declaration ops (but not the
    // send/recv ops)
    //   up above the split so they are visible across the region.
    // - Create the actual region computation op with a number of mlir regions
    // equal to
    //   the size of the localization mesh. Clone/move the basic block for
    //   each region. The region mesh should take in the live TENSOR values as
    //   arguments (propogated as basic block arguments) and should return the
    //   live out values as results.) Non-tensor values like tokens should
    //   remain from the higher namespace: region computation ops are not
    //   isolated.
    // - For each collective used in the region, decompose it into one token
    // per device using
    //   the SubmeshCollectiveParts op (placed outside the region
    //   computation), and replace uses of the token with the appropriate
    //   submesh token for the region computation op.
    // - Finally, replace the sharding attributes inside the region
    // computation and conver the types
    //   to the size of the local shards. Shardy has a pass
    //   ConvertGlobalToLocal (convert_global_to_local.cc)
    //    that does this part, so ideally we can reuse that logic rather than
    //    implementing.
  }

  for (PlannedLocalization &planned : plannedLocalizations) {
    // Step 2+ will mutate IR based on pre-extracted ranges/localizations.
    FailureOr<SplitBlocks> splitOr = splitPlannedRangeIntoBlocks(planned);
    if (failed(splitOr)) {
      return failure();
    }

    SplitBlocks split = *splitOr;
    Block *rangeBlock = split.range;

    auto parentLogicalMesh =
        meshComputationOp.getMesh().getDefiningOp<LogicalMeshOp>();
    if (!parentLogicalMesh) {
      meshComputationOp.emitError()
          << "expected mesh operand to be defined by distributed.LogicalMesh";
      return failure();
    }

    OpBuilder builder(split.pre, split.pre->end());
    llvm::SmallVector<Value> localizationAxes(planned.localization.begin(),
                                              planned.localization.end());
    auto localizedLogicalMesh = builder.create<LogicalMeshOp>(
        meshComputationOp.getLoc(), parentLogicalMesh.getPhysicalMeshAttr(),
        localizationAxes);

    llvm::SmallVector<CollectiveOp> hoistedCollectives =
        hoistMetadataOps(split.range, split.pre);

    Liveness livenessAnalysis(meshComputationOp);
    const Liveness::ValueSetT &liveInSet =
        livenessAnalysis.getLiveIn(rangeBlock);
    const Liveness::ValueSetT &liveOutSet =
        livenessAnalysis.getLiveOut(rangeBlock);

    llvm::SmallVector<Value> liveInsAll(liveInSet.begin(), liveInSet.end());
    llvm::SmallVector<Value> liveOutsAll(liveOutSet.begin(), liveOutSet.end());
    llvm::SmallVector<Value> liveIns = filterTensorValues(liveInsAll);
    llvm::SmallVector<Value> liveOuts = filterTensorValues(liveOutsAll);
    liveOuts = filterLiveOutsDefinedInRange(split.range, liveOuts);

    FailureOr<int64_t> submeshSize = localizedLogicalMesh.getMeshSize();
    if (failed(submeshSize)) {
      localizedLogicalMesh.emitError()
          << "failed to determine localized submesh size";
      return failure();
    }
    unsigned n_devices = static_cast<unsigned>(*submeshSize);

    llvm::SmallVector<Type> regionResultTypes;
    regionResultTypes.reserve(liveOuts.size());
    for (Value value : liveOuts) {
      regionResultTypes.push_back(value.getType());
    }

    builder.setInsertionPoint(split.pre, split.pre->end());
    auto regionComputation = builder.create<RegionComputationOp>(
        meshComputationOp.getLoc(), regionResultTypes,
        localizedLogicalMesh.getMesh(), liveIns, n_devices);

    FailureOr<llvm::DenseMap<Value, llvm::SmallVector<Value>>> deviceTokenMap =
        buildDeviceTokenMap(hoistedCollectives, localizedLogicalMesh.getMesh(),
                            n_devices, builder, meshComputationOp.getLoc());
    if (failed(deviceTokenMap)) {
      return failure();
    }

    cloneRangeIntoRegionComputationBodies(regionComputation, split.range,
                                          liveIns, liveOuts, *deviceTokenMap,
                                          meshComputationOp.getLoc());
    replaceRangeWithRegionComputation(split, regionComputation, liveOuts);
  }

  return success();
}

struct LocalizeDistributedPass
    : public enzyme::distributed::impl::LocalizeDistributedPassBase<
          LocalizeDistributedPass> {
  using LocalizeDistributedPassBase::LocalizeDistributedPassBase;

  void runOnOperation() override {
    if (failed(::mlir::enzyme::distributed::localizeMeshComputation(
            getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir
