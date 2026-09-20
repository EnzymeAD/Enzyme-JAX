#include "CollectiveOps.h"

#include "mlir/IR/PatternMatch.h"

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
static FailureOr<SmallVector<int64_t>>
computeDimensionCounts(RangeT valueTypes) {
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
    return op->emitOpError()
           << "requires " << ownerName << " size to match value count ("
           << shardings.getShardings().size() << " != " << llvm::size(dimCounts)
           << ")";
  }

  for (auto [valueIndex, sharding] :
       llvm::enumerate(shardings.getShardings())) {
    int64_t expectedDimCount = dimCounts[valueIndex];
    if (sharding.getDimPartitioningAxes().size() !=
        static_cast<size_t>(expectedDimCount)) {
      return op->emitOpError()
             << "requires " << ownerName << "[" << valueIndex
             << "] dim_partitioning_axes size to match value dimension count ("
             << sharding.getDimPartitioningAxes().size()
             << " != " << expectedDimCount << ")";
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
                 << axisIndex << " to be in range [0, " << partitioningAxisCount
                 << ")";
        }
      }
    }

    // unreduced_axes holds partitioning-axis indices (like
    // dim_partitioning_axes entries), not tensor-dim indices.
    for (int64_t axisIndex : sharding.getUnreducedAxes().asArrayRef()) {
      if (axisIndex < 0 || axisIndex >= partitioningAxisCount) {
        return op->emitOpError()
               << "requires " << ownerName << "[" << valueIndex
               << "] unreduced_axes index " << axisIndex
               << " to be in range [0, " << partitioningAxisCount << ")";
      }
    }
  }

  return success();
}

static LogicalResult verifyIndexedShardingPerValueHasNoUnreducedAxes(
    Operation *op, IndexedTensorShardingPerValueAttr shardings,
    StringRef ownerName) {
  for (auto [valueIndex, sharding] :
       llvm::enumerate(shardings.getShardings())) {
    if (!sharding.getUnreducedAxes().empty()) {
      return op->emitOpError() << "requires " << ownerName << "[" << valueIndex
                               << "] to have no unreduced axes";
    }
  }
  return success();
}

static LogicalResult
inferTensorViewCastResultType(MLIRContext *context,
                              std::optional<Location> location, Value input,
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
  auto disjoint = axis::areFactorGroupsDisjoint(partitioningAxes);
  if (failed(disjoint)) {
    return castOp.emitOpError()
           << "requires partitioning-axis factor groups to be produced by "
              "axis.product";
  }
  if (!*disjoint) {
    return castOp.emitOpError()
           << "requires partitioning-axis factor groups to be disjoint";
  }
  return success();
}

// Folds a cast that round-trips straight back through its own opposite
// (DistributedCastGlobalToLocalOp<-DistributedCastLocalToGlobalOp or vice
// versa) using the exact same partitioning-axis operands both times. Both
// casts are pure relabeling with no data movement (see
// CanonicalizeShardedFactorOrderPass's top-of-file "same materialization"
// invariant), so reinterpreting a value one way and immediately back the
// other -- declaring the identical factor decomposition each time --
// reproduces the same materialization exactly. Folds to an
// AnchorPartitioningOp rather than bypassing straight to the original
// value: the binding still needs to be discoverable at this exact edge.
// Requires literal operand equality (not just axis-set equivalence) so this
// never has to reason about factor-order equivalence itself; a mismatched pair
// (e.g. one leg canonically reordered by CanonicalizeShardedFactorOrderPass) is
// deliberately left alone.
template <typename ThisOp, typename OppositeOp>
struct FoldCastRoundTrip : public OpRewritePattern<ThisOp> {
  using OpRewritePattern<ThisOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThisOp castOp,
                                PatternRewriter &rewriter) const override {
    auto producer = castOp.getInput().template getDefiningOp<OppositeOp>();
    if (!producer) {
      return failure();
    }
    if (!llvm::equal(producer.getPartitioningAxes(),
                     castOp.getPartitioningAxes())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<AnchorPartitioningOp>(
        castOp, producer.getInput(), producer.getPartitioningAxes());
    return success();
  }
};

// Folds an AnchorPartitioningOp whose own input is already produced by any
// PartitioningAnchorOpInterface op declaring the exact same
// partitioning_axes for its result -- the anchor is then purely redundant
// (the identical binding is already discoverable directly at its own
// input), so replace it with that input value itself. Only handles the case
// where every dimension of the producer's own binding resolves to exactly
// one value (true for every current implementer except a composite
// DistributedKernelOp slot, which this deliberately leaves alone rather
// than reasoning about multi-value-per-dimension equivalence).
struct FoldRedundantAnchor : public OpRewritePattern<AnchorPartitioningOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AnchorPartitioningOp anchorOp,
                                PatternRewriter &rewriter) const override {
    auto producer = dyn_cast_or_null<PartitioningAnchorOpInterface>(
        anchorOp.getInput().getDefiningOp());
    if (!producer) {
      return failure();
    }
    auto producerResult = cast<OpResult>(anchorOp.getInput());
    auto producerBinding = producer.getBindingInfoForResult(producerResult);
    if (failed(producerBinding)) {
      return failure();
    }
    SmallVector<Value> flatProducerAxes;
    flatProducerAxes.reserve(producerBinding->size());
    for (ArrayRef<Value> dimValues : *producerBinding) {
      if (dimValues.size() != 1) {
        return failure();
      }
      flatProducerAxes.push_back(dimValues.front());
    }
    if (!llvm::equal(flatProducerAxes, anchorOp.getPartitioningAxes())) {
      return failure();
    }
    rewriter.replaceOp(anchorOp, anchorOp.getInput());
    return success();
  }
};

// Shared PartitioningAnchorOpInterface implementation for every op whose
// own partitioning_axes operand is already one factor-group value per
// tensor dimension (both Cast ops, and AnchorPartitioningOp) -- as opposed
// to DistributedKernelOp, which addresses a possibly-composite
// decomposition by index into a shared list instead.
template <typename OpTy>
static FailureOr<SmallVector<SmallVector<Value>>>
singleValueBindingInfo(OpTy op, unsigned operandOrResultIndex) {
  if (operandOrResultIndex != 0) {
    return failure();
  }
  SmallVector<SmallVector<Value>> result;
  result.reserve(op.getPartitioningAxes().size());
  for (Value axis : op.getPartitioningAxes()) {
    result.push_back({axis});
  }
  return result;
}

// Shared PartitioningAnchorOpInterface implementation for DistributedKernelOp
// and DistributedCallOp:
// resolves `shardings`[valueIndex]'s dim_partitioning_axes (lists of indices
// into the kernel's own shared partitioning_axes) into the actual
// FactorGroupType values at those indices, one inner list per tensor
// dimension.
template <typename OpTy>
static FailureOr<SmallVector<SmallVector<Value>>>
kernelBindingInfo(OpTy kernelOp, IndexedTensorShardingPerValueAttr shardings,
                  unsigned valueIndex) {
  ArrayRef<IndexedTensorShardingAttr> perValue = shardings.getShardings();
  if (valueIndex >= perValue.size()) {
    return failure();
  }
  ValueRange partitioningAxes = kernelOp.getPartitioningAxes();
  SmallVector<SmallVector<Value>> result;
  result.reserve(perValue[valueIndex].getDimPartitioningAxes().size());
  for (DenseI64ArrayAttr dimAxes :
       perValue[valueIndex].getDimPartitioningAxes()) {
    SmallVector<Value> dimValues;
    dimValues.reserve(dimAxes.size());
    for (int64_t idx : dimAxes.asArrayRef()) {
      if (idx < 0 || idx >= static_cast<int64_t>(partitioningAxes.size())) {
        return failure();
      }
      dimValues.push_back(partitioningAxes[idx]);
    }
    result.push_back(std::move(dimValues));
  }
  return result;
}

// Checks the real relationship DistributedKernelOp::verify() requires
// between one operand/result (`localValue`, at this op's own LOCAL scope)
// and its corresponding block-argument/yielded-value type
// (`globalScopeType`, at GLOBAL scope): the two must have the same rank, be
// identical wherever `binding` names no factors for a dimension, and
// otherwise globalScopeType[d] == localValue's shape[d] * (product of
// binding[d]'s raw factor extents, EXCLUDING ReplicationAxisType provenance
// -- a replicated axis is the same full extent on every device, so it never
// divides the local size down from the global one, unlike a
// DeviceLocalAxis, which does count here even though it's never physically
// mesh-sharded). Unresolvable factor extents are skipped rather than
// flagged (same "don't fail what we can't resolve" stance as
// resolveCurrentSharding elsewhere in this dialect) -- this check is only
// as strong as the axis-algebra info currently available. Rank itself is
// NOT similarly best-effort: partitioning can never add or remove a
// dimension, so a rank mismatch between the local and global views of the
// same operand/result is always a real bug (e.g. a pass that rebuilt one
// side's type but not the other), never a case this check should stay
// silent about.
static LogicalResult
checkLocalGlobalBinding(Operation *kernelOp, Value localValue,
                        Type globalScopeType,
                        ArrayRef<SmallVector<Value>> binding, StringRef what) {
  auto localType = dyn_cast<RankedTensorType>(localValue.getType());
  auto globalType = dyn_cast<RankedTensorType>(globalScopeType);
  if (!localType || !globalType) {
    return success();
  }
  if (localType.getRank() != globalType.getRank()) {
    return kernelOp->emitOpError()
           << what << "'s local-scope type " << localType
           << " and global-scope type " << globalType
           << " have different ranks -- partitioning can only divide an "
              "existing dimension's size, never add or remove one, so this "
              "always means some pass rebuilt one side's type without "
              "keeping the other in sync";
  }
  if (static_cast<int64_t>(binding.size()) != localType.getRank()) {
    return kernelOp->emitOpError()
           << what << "'s resolved binding has " << binding.size()
           << " dimension(s) but its local-scope type " << localType
           << " has rank " << localType.getRank();
  }
  for (auto [dim, dimValues] : llvm::enumerate(binding)) {
    int64_t extent = 1;
    for (Value v : dimValues) {
      auto factorGroup = dyn_cast<TypedValue<axis::FactorGroupType>>(v);
      if (!factorGroup) {
        continue;
      }
      auto rawFactors = axis::getProductProvenanceFactors(factorGroup);
      if (failed(rawFactors)) {
        continue;
      }
      for (auto factor : *rawFactors) {
        auto provenance = axis::getFactorProvenanceAxis(factor);
        if (failed(provenance) ||
            isa<ReplicationAxisType>(provenance->getType())) {
          continue;
        }
        extent *= static_cast<int64_t>(axis::getFactorExtent(factor));
      }
    }
    int64_t localDim = localType.getDimSize(dim);
    int64_t globalDim = globalType.getDimSize(dim);
    if (extent == 1) {
      continue; // nothing declared (or unresolvable) for this dimension -- not
                // this check's job to require anything.
    }
    if (globalDim != localDim * extent) {
      return kernelOp->emitOpError()
             << what << " dimension " << dim << ": global-scope size ("
             << globalDim << ") does not equal local-scope size (" << localDim
             << ") times its declared partitioning extent (" << extent << ")";
    }
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
          getOperation(), outputShardings, *outputDimCounts, "output_shardings",
          partitioningAxisCount))) {
    return failure();
  }
  if (failed(verifyIndexedShardingPerValueHasNoUnreducedAxes(
          getOperation(), argumentShardings, "argument_shardings"))) {
    return failure();
  }
  return success();
}

LogicalResult DistributedCallOp::verify() {
  int64_t partitioningAxisCount =
      static_cast<int64_t>(getPartitioningAxes().size());

  SmallVector<int64_t> argumentDimCounts;
  for (Value argument : getArguments()) {
    FailureOr<int64_t> dimCount = getValueDimensionCount(argument.getType());
    if (failed(dimCount)) {
      return emitOpError() << "failed to compute argument dimension counts";
    }
    argumentDimCounts.push_back(*dimCount);
  }
  SmallVector<int64_t> outputDimCounts;
  for (Type resultType : getResultTypes()) {
    FailureOr<int64_t> dimCount = getValueDimensionCount(resultType);
    if (failed(dimCount)) {
      return emitOpError() << "failed to compute result dimension counts";
    }
    outputDimCounts.push_back(*dimCount);
  }

  if (failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), getArgumentShardings(), argumentDimCounts,
          "argument_shardings", partitioningAxisCount)) ||
      failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), getOutputShardings(), outputDimCounts,
          "output_shardings", partitioningAxisCount)) ||
      failed(verifyIndexedShardingPerValueHasNoUnreducedAxes(
          getOperation(), getArgumentShardings(), "argument_shardings"))) {
    return failure();
  }
  return success();
}

LogicalResult
DistributedCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = symbolTable.lookupNearestSymbolFrom<DistributedFunctionOp>(
      *this, getCalleeAttr());
  if (!callee) {
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a distributed function";
  }
  FunctionType type = callee.getFunctionType();
  if (type.getNumInputs() != getArguments().size() ||
      type.getNumResults() != getNumResults()) {
    return emitOpError() << "operand and result counts must match the "
                            "signature of '"
                         << getCallee() << "'";
  }

  // The callee's signature is at global scope: each local operand/result must
  // relate to it through this call's declared partitioning axes.
  for (auto [idx, arg] : llvm::enumerate(getArguments())) {
    auto binding = getBindingInfoForOperand(getOperation()->getOpOperand(idx));
    if (failed(binding)) {
      continue;
    }
    if (failed(checkLocalGlobalBinding(getOperation(), arg, type.getInput(idx),
                                       *binding, "operand"))) {
      return failure();
    }
  }
  for (auto [idx, result] : llvm::enumerate(getResults())) {
    auto binding = getBindingInfoForResult(cast<OpResult>(result));
    if (failed(binding)) {
      continue;
    }
    if (failed(checkLocalGlobalBinding(
            getOperation(), result, type.getResult(idx), *binding, "result"))) {
      return failure();
    }
  }
  return success();
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedCallOp::getBindingInfoForOperand(OpOperand &operand) {
  return kernelBindingInfo(*this, getArgumentShardings(),
                           operand.getOperandNumber());
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedCallOp::getBindingInfoForResult(OpResult result) {
  return kernelBindingInfo(*this, getOutputShardings(),
                           result.getResultNumber());
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
          getOperation(), outputShardings, outputDimCounts, "output_shardings",
          partitioningAxisCount))) {
    return failure();
  }
  if (failed(verifyIndexedShardingPerValueHasNoUnreducedAxes(
          getOperation(), argumentShardings, "argument_shardings"))) {
    return failure();
  }

  // The real local(operand)<->global(block-arg) relationship -- see this
  // op's own doc comment. Checked here, unconditionally, because it's meant
  // to hold continuously, not just after some specific pass has run: every
  // pass that changes a factor's classification or extent is responsible
  // for keeping the operand/block-arg (and result/yield) pair in sync at
  // the same time, and this is what catches it immediately if one ever
  // doesn't.
  //
  // Both count checks below are hard errors, not best-effort skips: the
  // block's own argument count and the yield's own operand count are
  // structural properties of this op (by construction, one block argument
  // per operand, one yielded value per result), so a mismatch here always
  // means some pass edited one list -- operands, block arguments, results,
  // or the yield -- without keeping the other three in sync, exactly the
  // class of bug this whole check exists to catch immediately rather than
  // let surface as a confusing shape error somewhere downstream.
  auto &bodyBlock = getBody().front();
  if (getArguments().size() != bodyBlock.getNumArguments()) {
    return emitOpError() << "has " << getArguments().size()
                         << " operand(s) but its body block has "
                         << bodyBlock.getNumArguments() << " argument(s)";
  }
  for (auto [idx, arg] : llvm::enumerate(getArguments())) {
    auto binding = getBindingInfoForOperand(getOperation()->getOpOperand(idx));
    if (failed(binding)) {
      continue;
    }
    if (failed(checkLocalGlobalBinding(*this, arg,
                                       bodyBlock.getArgument(idx).getType(),
                                       *binding, "operand"))) {
      return failure();
    }
  }

  auto yieldOp = dyn_cast<DistributedYieldOp>(bodyBlock.getTerminator());
  if (!yieldOp) {
    return emitOpError() << "body block must be terminated by a "
                            "distributed.DistributedYield";
  }
  if (yieldOp.getReturns().size() != getResults().size()) {
    return emitOpError() << "has " << getResults().size()
                         << " result(s) but its body yields "
                         << yieldOp.getReturns().size() << " value(s)";
  }
  for (auto [idx, result] : llvm::enumerate(getResults())) {
    auto binding = getBindingInfoForResult(cast<OpResult>(result));
    if (failed(binding)) {
      continue;
    }
    if (failed(checkLocalGlobalBinding(*this, result,
                                       yieldOp.getReturns()[idx].getType(),
                                       *binding, "result"))) {
      return failure();
    }
  }

  return success();
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedKernelOp::getBindingInfoForOperand(OpOperand &operand) {
  return kernelBindingInfo(*this, getArgumentShardings(),
                           operand.getOperandNumber());
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedKernelOp::getBindingInfoForResult(OpResult result) {
  return kernelBindingInfo(*this, getOutputShardings(),
                           result.getResultNumber());
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

void DistributedCastGlobalToLocalOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldCastRoundTrip<DistributedCastGlobalToLocalOp,
                                DistributedCastLocalToGlobalOp>>(context);
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedCastGlobalToLocalOp::getBindingInfoForOperand(OpOperand &operand) {
  return singleValueBindingInfo(*this, operand.getOperandNumber());
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedCastGlobalToLocalOp::getBindingInfoForResult(OpResult result) {
  return singleValueBindingInfo(*this, result.getResultNumber());
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

void DistributedCastLocalToGlobalOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldCastRoundTrip<DistributedCastLocalToGlobalOp,
                                DistributedCastGlobalToLocalOp>>(context);
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedCastLocalToGlobalOp::getBindingInfoForOperand(OpOperand &operand) {
  return singleValueBindingInfo(*this, operand.getOperandNumber());
}

FailureOr<SmallVector<SmallVector<Value>>>
DistributedCastLocalToGlobalOp::getBindingInfoForResult(OpResult result) {
  return singleValueBindingInfo(*this, result.getResultNumber());
}

// AnchorPartitioningOp is a pure re-anchor: rank must match
// partitioning_axes.size() (same structural check as the Cast ops), and
// factor groups must be pairwise disjoint. Type equality between input and
// output is already enforced structurally by the AllTypesMatch trait, so
// this verifier doesn't need to (and can't meaningfully, given growth can
// legitimately make the two diverge via forward propagation -- see this
// op's own doc comment) check shapes itself.
LogicalResult AnchorPartitioningOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    return emitOpError() << "requires a ranked tensor input";
  }
  if (getPartitioningAxes().size() !=
      static_cast<size_t>(inputType.getRank())) {
    return emitOpError() << "requires one partitioning axis per tensor "
                         << "dimension";
  }
  auto partitioningAxes = axis::castTypedValueList<axis::FactorGroupType>(
      getPartitioningAxes(), "FactorGroupType");
  auto disjoint = axis::areFactorGroupsDisjoint(partitioningAxes);
  if (failed(disjoint)) {
    return emitOpError() << "requires partitioning-axis factor groups to be "
                            "produced by axis.product";
  }
  if (!*disjoint) {
    return emitOpError()
           << "requires partitioning-axis factor groups to be disjoint";
  }
  return success();
}

void AnchorPartitioningOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldRedundantAnchor>(context);
}

FailureOr<SmallVector<SmallVector<Value>>>
AnchorPartitioningOp::getBindingInfoForOperand(OpOperand &operand) {
  return singleValueBindingInfo(*this, operand.getOperandNumber());
}

FailureOr<SmallVector<SmallVector<Value>>>
AnchorPartitioningOp::getBindingInfoForResult(OpResult result) {
  return singleValueBindingInfo(*this, result.getResultNumber());
}

LogicalResult DistributedManualComputationOp::verify() {
  auto kernelOp = getOperation()->getParentOfType<DistributedKernelOp>();
  if (!kernelOp) {
    return emitOpError()
           << "requires an enclosing distributed.DistributedKernel";
  }
  ValueRange partitioningAxes = kernelOp.getPartitioningAxes();
  int64_t partitioningAxisCount = static_cast<int64_t>(partitioningAxes.size());

  // manual_axes is the caller's own choice of which slots Shardy should
  // divide down for this op's region. A DeviceLocalAxisType/ReplicationAxisType
  // factor is always excluded from LowerKernels.cpp's real mesh-axis sizing
  // (parallelismPerDim), so marking one manual would just divide by 1 -- a
  // meaningless no-op that almost certainly indicates a caller bug (the axis
  // that needs manual, real division is the sharded one, not the local one).
  llvm::SmallDenseSet<int64_t> manualAxisSet;
  for (int64_t axisIndex : getManualAxes()) {
    if (axisIndex < 0 || axisIndex >= partitioningAxisCount) {
      return emitOpError() << "requires manual_axes index " << axisIndex
                           << " to be in range [0, " << partitioningAxisCount
                           << ")";
    }
    if (!manualAxisSet.insert(axisIndex).second) {
      return emitOpError() << "requires manual_axes to be pairwise distinct ("
                           << axisIndex << " repeated)";
    }
    auto factorGroup =
        cast<TypedValue<axis::FactorGroupType>>(partitioningAxes[axisIndex]);
    auto factors = axis::getProductProvenanceFactors(factorGroup);
    if (failed(factors)) {
      return emitOpError() << "requires manual_axes[" << axisIndex
                           << "] to be produced by axis.product";
    }
    for (auto factor : *factors) {
      auto provenance = axis::getFactorProvenanceAxis(factor);
      if (failed(provenance)) {
        return emitOpError() << "requires manual_axes[" << axisIndex
                             << "]'s factors to have a resolvable provenance";
      }
      if (isa<DeviceLocalAxisType, ReplicationAxisType>(
              provenance->getType())) {
        return emitOpError()
               << "requires manual_axes[" << axisIndex
               << "] to not be a DeviceLocalAxisType/ReplicationAxisType "
                  "factor (never really divided by LowerKernels.cpp)";
      }
    }
  }

  auto argumentShardings = getArgumentShardings();
  auto outputShardings = getOutputShardings();
  FailureOr<SmallVector<int64_t>> inputDimCounts =
      computeDimensionCounts(getInputs().getTypes());
  FailureOr<SmallVector<int64_t>> outputDimCounts =
      computeDimensionCounts(getOutputs().getTypes());
  if (failed(inputDimCounts) || failed(outputDimCounts)) {
    return emitOpError() << "failed to compute input/output dimension counts";
  }
  if (failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), argumentShardings, *inputDimCounts,
          "argument_shardings", partitioningAxisCount)) ||
      failed(verifyIndexedShardingPerValueAgainstDimensionRanges(
          getOperation(), outputShardings, *outputDimCounts, "output_shardings",
          partitioningAxisCount))) {
    return failure();
  }

  // Same grammar Shardy's own sdy.manual_computation requires: a manual axis
  // must precede any non-manual one within its own dimension's index list.
  // Nothing else is required of a non-manual axis -- it's fine for one to
  // also be device-distributed (Shardy is free to treat it as an ordinary
  // sharded/free axis; this op just doesn't divide it down further itself).
  auto checkManualMajormost = [&](IndexedTensorShardingPerValueAttr shardings,
                                  StringRef ownerName) -> LogicalResult {
    for (auto [valueIndex, sharding] :
         llvm::enumerate(shardings.getShardings())) {
      for (auto [dimIndex, dimAxes] :
           llvm::enumerate(sharding.getDimPartitioningAxes())) {
        bool sawNonManual = false;
        for (int64_t axisIndex : dimAxes.asArrayRef()) {
          if (!manualAxisSet.contains(axisIndex)) {
            sawNonManual = true;
          } else if (sawNonManual) {
            return emitOpError()
                   << "requires manual axes to precede non-manual axes in "
                   << ownerName << "[" << valueIndex
                   << "]'s dim_partitioning_axes[" << dimIndex << "]";
          }
        }
      }
    }
    return success();
  };
  if (failed(checkManualMajormost(argumentShardings, "argument_shardings")) ||
      failed(checkManualMajormost(outputShardings, "output_shardings"))) {
    return failure();
  }

  // Each manual-axis dimension of an input/output must be evenly divisible by
  // the product of its manual axes' extents, with the region's corresponding
  // block-argument/yield-operand dimension equal to that quotient; every
  // other dimension is unchanged between the outer (global) and inner
  // (local) view.
  auto &bodyBlock = getBody().front();
  auto yieldOp = dyn_cast<DistributedYieldOp>(bodyBlock.getTerminator());
  if (!yieldOp) {
    return emitOpError() << "requires its body to be terminated by "
                            "distributed.DistributedYield";
  }
  auto checkLocalShape = [&](Value outer, Type localType,
                             IndexedTensorShardingAttr sharding,
                             StringRef what) -> LogicalResult {
    auto outerType = dyn_cast<RankedTensorType>(outer.getType());
    auto localRankedType = dyn_cast<RankedTensorType>(localType);
    if (!outerType || !localRankedType) {
      return success();
    }
    SmallVector<int64_t> expectedLocalShape(outerType.getShape());
    for (auto [dim, dimAxes] :
         llvm::enumerate(sharding.getDimPartitioningAxes())) {
      if (dim >= expectedLocalShape.size()) {
        break;
      }
      int64_t divisor = 1;
      for (int64_t axisIndex : dimAxes.asArrayRef()) {
        if (!manualAxisSet.contains(axisIndex)) {
          continue;
        }
        auto factorGroup = cast<TypedValue<axis::FactorGroupType>>(
            partitioningAxes[axisIndex]);
        FailureOr<uint64_t> extent = axis::getFactorGroupExtent(factorGroup);
        if (failed(extent)) {
          return emitOpError() << "requires manual_axes[" << axisIndex
                               << "] to have a resolvable extent";
        }
        divisor *= static_cast<int64_t>(*extent);
      }
      if (divisor == 1) {
        continue;
      }
      if (expectedLocalShape[dim] % divisor != 0) {
        return emitOpError()
               << what << " dimension " << dim
               << " is not evenly divisible by its manual axes' extents";
      }
      expectedLocalShape[dim] /= divisor;
    }
    if (localRankedType.getShape() != ArrayRef<int64_t>(expectedLocalShape)) {
      return emitOpError()
             << what
             << "'s local (region) type does not match its global "
                "type divided by its manual axes' extents";
    }
    return success();
  };

  if (getInputs().size() != bodyBlock.getNumArguments() ||
      argumentShardings.getShardings().size() != getInputs().size()) {
    return emitOpError() << "requires one region block argument and one "
                            "argument_shardings entry per input";
  }
  for (auto [input, blockArg, sharding] :
       llvm::zip_equal(getInputs(), bodyBlock.getArguments(),
                       argumentShardings.getShardings())) {
    if (failed(checkLocalShape(input, blockArg.getType(), sharding,
                               "an input's"))) {
      return failure();
    }
  }

  if (getOutputs().size() != yieldOp.getReturns().size() ||
      outputShardings.getShardings().size() != getOutputs().size()) {
    return emitOpError() << "requires one yielded value and one "
                            "output_shardings entry per output";
  }
  for (auto [output, yielded, sharding] :
       llvm::zip_equal(getOutputs(), yieldOp.getReturns(),
                       outputShardings.getShardings())) {
    if (failed(checkLocalShape(output, yielded.getType(), sharding,
                               "an output's"))) {
      return failure();
    }
  }

  return success();
}

} // namespace mlir::enzyme::distributed
