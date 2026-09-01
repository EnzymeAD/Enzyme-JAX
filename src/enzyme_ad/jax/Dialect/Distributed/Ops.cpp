#include "CollectiveOps.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

// Central emission point for generated distributed op class definitions.
// Keep this in a dedicated file so op definitions do not depend on any
// specific op implementation unit remaining present.
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"

namespace mlir::enzyme::distributed {

namespace {

static FailureOr<int64_t> getValueDimensionCount(Type valueType) {
  if (auto shapedType = dyn_cast<ShapedType>(valueType)) {
    return static_cast<int64_t>(shapedType.getRank());
  }
  return static_cast<int64_t>(0);
}

template <typename RangeT>
static FailureOr<SmallVector<int64_t>> computeDimensionCounts(RangeT valueTypes) {
  SmallVector<int64_t> dimCounts;
  dimCounts.reserve(llvm::size(valueTypes));
  for (Type valueType : valueTypes) {
    FailureOr<int64_t> dimCount = getValueDimensionCount(valueType);
    if (failed(dimCount)) {
      return failure();
    }
    dimCounts.push_back(*dimCount);
  }
  return dimCounts;
}

template <typename DimCountRangeT>
static LogicalResult verifyIndexedShardingPerValueAgainstDimensionRanges(
    Operation *op, IndexedTensorShardingPerValueAttr shardings,
    DimCountRangeT dimCounts, StringRef ownerName,
    int64_t partitioningAxisCount) {
  if (shardings.getShardings().size() != llvm::size(dimCounts)) {
    return op->emitOpError() << "requires " << ownerName
                             << " size to match value count ("
                             << shardings.getShardings().size() << " != "
                             << llvm::size(dimCounts) << ")";
  }

  for (auto [valueIndex, sharding] : llvm::enumerate(shardings.getShardings())) {
    int64_t expectedDimCount = dimCounts[valueIndex];
    if (sharding.getDimPartitioningAxes().size() !=
        static_cast<size_t>(expectedDimCount)) {
      return op->emitOpError()
             << "requires " << ownerName << "[" << valueIndex
             << "] dim_partitioning_axes size to match value dimension count ("
             << sharding.getDimPartitioningAxes().size() << " != "
             << expectedDimCount << ")";
    }

    for (auto [dimIndex, dimPartitionAxes] :
         llvm::enumerate(sharding.getDimPartitioningAxes())) {
      if (static_cast<int64_t>(dimIndex) >= expectedDimCount) {
        return op->emitOpError()
               << "requires " << ownerName << "[" << valueIndex
               << "] dim_partitioning_axes index " << dimIndex
               << " to be in range [0, " << expectedDimCount << ")";
      }

      for (int64_t axisIndex : dimPartitionAxes.asArrayRef()) {
        if (axisIndex < 0 || axisIndex >= partitioningAxisCount) {
          return op->emitOpError()
                 << "requires " << ownerName << "[" << valueIndex
                 << "] dim_partitioning_axes[" << dimIndex << "] index "
                 << axisIndex << " to be in range [0, "
                 << partitioningAxisCount << ")";
        }
      }
    }

    for (int64_t dimIndex : sharding.getUnreducedAxes().asArrayRef()) {
      if (dimIndex < 0 || dimIndex >= expectedDimCount) {
        return op->emitOpError()
               << "requires " << ownerName << "[" << valueIndex
               << "] unreduced_axes index " << dimIndex
               << " to be in range [0, " << expectedDimCount << ")";
      }
    }
  }

  return success();
}

static LogicalResult verifyIndexedShardingPerValueHasNoUnreducedAxes(
    Operation *op, IndexedTensorShardingPerValueAttr shardings,
    StringRef ownerName) {
  for (auto [valueIndex, sharding] : llvm::enumerate(shardings.getShardings())) {
    if (!sharding.getUnreducedAxes().empty()) {
      return op->emitOpError() << "requires " << ownerName << "[" << valueIndex
                               << "] to have no unreduced axes";
    }
  }
  return success();
}

static LogicalResult inferTensorViewCastResultType(
    MLIRContext *context, std::optional<Location> location, Value input,
    ValueRange partitioningAxes, bool globalToLocal,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape()) {
    if (location) {
      emitError(*location)
          << "requires a statically shaped ranked tensor input";
    }
    return failure();
  }
  if (partitioningAxes.size() != static_cast<size_t>(inputType.getRank())) {
    if (location) {
      emitError(*location) << "requires one partitioning axis per tensor "
                           << "dimension";
    }
    return failure();
  }

  SmallVector<int64_t> outputShape;
  outputShape.reserve(inputType.getRank());
  for (auto [dim, axis] : llvm::enumerate(partitioningAxes)) {
    auto factorGroup = dyn_cast<TypedValue<axis::FactorGroupType>>(axis);
    if (!factorGroup) {
      if (location) {
        emitError(*location) << "requires partitioning axes to have "
                             << "FactorGroupType";
      }
      return failure();
    }
    FailureOr<uint64_t> extent = axis::getFactorGroupExtent(factorGroup);
    if (failed(extent) || *extent == 0) {
      if (location) {
        emitError(*location) << "requires partitioning axes to have "
                             << "resolvable positive extents";
      }
      return failure();
    }

    int64_t inputDim = inputType.getDimSize(dim);
    if (globalToLocal) {
      if (inputDim % static_cast<int64_t>(*extent) != 0) {
        if (location) {
          emitError(*location) << "requires each global tensor dimension to "
                               << "be divisible by its partitioning extent";
        }
        return failure();
      }
      outputShape.push_back(inputDim / static_cast<int64_t>(*extent));
    } else {
      outputShape.push_back(inputDim * static_cast<int64_t>(*extent));
    }
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(outputShape, inputType.getElementType()));
  return success();
}

template <typename CastOp>
static LogicalResult verifyTensorViewCast(CastOp castOp, bool globalToLocal) {
  SmallVector<Type> inferredResultTypes;
  if (failed(inferTensorViewCastResultType(
          castOp.getContext(), castOp.getLoc(), castOp.getInput(),
          castOp.getPartitioningAxes(), globalToLocal, inferredResultTypes))) {
    return failure();
  }
  if (castOp.getOutput().getType() != inferredResultTypes.front()) {
    return castOp.emitOpError()
           << "requires result type to match the input shape and "
              "partitioning-axis extents";
  }

  auto partitioningAxes = axis::castTypedValueList<axis::FactorGroupType>(
      castOp.getPartitioningAxes(), "FactorGroupType");
  if (!axis::areFactorGroupsDisjoint(partitioningAxes)) {
    return castOp.emitOpError()
           << "requires partitioning-axis factor groups to be disjoint";
  }
  return success();
}

} // namespace

LogicalResult DistributedFunctionOp::verify() {
  auto functionType = getFunctionType();
  auto argumentShardings = getArgumentShardings();
  auto outputShardings = getOutputShardings();
  int64_t partitioningAxisCount =
      static_cast<int64_t>(getPartitioningAxes().size());

  FailureOr<SmallVector<int64_t>> argumentDimCounts =
      computeDimensionCounts(functionType.getInputs());
  if (failed(argumentDimCounts)) {
    return emitOpError() << "failed to compute argument dimension counts";
  }
  FailureOr<SmallVector<int64_t>> outputDimCounts =
      computeDimensionCounts(functionType.getResults());
  if (failed(outputDimCounts)) {
    return emitOpError() << "failed to compute result dimension counts";
  }

  if (failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), argumentShardings, *argumentDimCounts,
          "argument_shardings", partitioningAxisCount))) {
    return failure();
  }
  if (failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), outputShardings, *outputDimCounts,
          "output_shardings", partitioningAxisCount))) {
    return failure();
  }
  if (failed(verifyIndexedShardingPerValueHasNoUnreducedAxes(
          getOperation(), argumentShardings, "argument_shardings"))) {
    return failure();
  }
  return success();
}

LogicalResult DistributedKernelOp::verify() {
  auto argumentShardings = getArgumentShardings();
  auto outputShardings = getOutputShardings();
  int64_t partitioningAxisCount =
      static_cast<int64_t>(getPartitioningAxes().size());

  SmallVector<int64_t> argumentDimCounts;
  argumentDimCounts.reserve(getArguments().size());
  for (Value argument : getArguments()) {
    FailureOr<int64_t> dimCount = getValueDimensionCount(argument.getType());
    if (failed(dimCount)) {
      return emitOpError() << "failed to compute argument dimension counts";
    }
    argumentDimCounts.push_back(*dimCount);
  }

  SmallVector<int64_t> outputDimCounts;
  outputDimCounts.reserve(getReturns().size());
  for (Type resultType : getResultTypes()) {
    FailureOr<int64_t> dimCount = getValueDimensionCount(resultType);
    if (failed(dimCount)) {
      return emitOpError() << "failed to compute result dimension counts";
    }
    outputDimCounts.push_back(*dimCount);
  }

  if (failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), argumentShardings, argumentDimCounts,
          "argument_shardings", partitioningAxisCount))) {
    return failure();
  }
  if (failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), outputShardings, outputDimCounts,
          "output_shardings", partitioningAxisCount))) {
    return failure();
  }
  if (failed(verifyIndexedShardingPerValueHasNoUnreducedAxes(
          getOperation(), argumentShardings, "argument_shardings"))) {
    return failure();
  }
  return success();
}

LogicalResult DistributedCastGlobalToLocalOp::verify() {
  return verifyTensorViewCast(*this, /*globalToLocal=*/true);
}

LogicalResult DistributedCastGlobalToLocalOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DistributedCastGlobalToLocalOpAdaptor adaptor(operands, attributes,
                                                 properties, regions);
  return inferTensorViewCastResultType(
      context, location, adaptor.getInput(), adaptor.getPartitioningAxes(),
      /*globalToLocal=*/true, inferredReturnTypes);
}

LogicalResult DistributedCastLocalToGlobalOp::verify() {
  return verifyTensorViewCast(*this, /*globalToLocal=*/false);
}

LogicalResult DistributedCastLocalToGlobalOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DistributedCastLocalToGlobalOpAdaptor adaptor(operands, attributes,
                                                 properties, regions);
  return inferTensorViewCastResultType(
      context, location, adaptor.getInput(), adaptor.getPartitioningAxes(),
      /*globalToLocal=*/false, inferredReturnTypes);
}

} // namespace mlir::enzyme::distributed
