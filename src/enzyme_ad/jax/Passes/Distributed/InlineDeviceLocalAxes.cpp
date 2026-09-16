#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

/**
 * A distributed.DeviceLocalAxis factor marks an axis the search process
 * decided to serialize within one device rather than shard across the mesh.
 * "Inlining" one means replacing that abstract marker with a concrete tensor
 * dimension of the same extent, so LowerKernelsPass sees real shapes. Global
 * tensor shapes remain the same, while local views get expanded to reflect
 * the new knowledge about which axes are device-local.
 *
 * Assumptions this pass relies on:
 *  - Inside a kernel body everything is at "global" size; DeviceLocalAxis
 *    factors there are invisible (LowerKernelsPass's isShardableFactor
 *    already ignores them) and never need touching.
 *  - CanonicalizeShardedFactorOrderPass guarantees DeviceLocalAxis factors
 *    are contiguous and minor-most per dimension, so growing a dimension by
 *    their combined extent is always an unambiguous multiply.
 *  - This pass only grows a value an op *owns* (its own result); operands
 *    are never grown directly. This means we expect local tensor shapes to
 *    be a result of one of our expected operations.
 *
 * Cases handled, one per op kind:
 *  - DistributedCastGlobalToLocalOp: grows its own result (the local side)
 *    per dimension, and drops those factors from partitioning_axes.
 *  - DistributedCastLocalToGlobalOp: symmetric, but its local side is an
 *    operand (someone else's result), so only partitioning_axes is
 *    rewritten.
 *  - DistributedKernelOp: grows its own external result types. Operand
 *    types are otherwise left alone (see the propagation assumption above),
 *    with one exception: a plain UnrealizedConversionCastOp feeding an
 *    operand (the placeholder ClusterDistributedKernels.cpp leaves at a
 *    kernel's own argument boundary) is grown directly, since it carries no
 *    shape information of its own to propagate from anywhere else.
 *  - DistributedCollectiveOp:
 *    - input_mesh/output_mesh factors are dropped: that part of the index
 *      space now lives on the tensor's own (already-grown) dimension
 *      instead of the mesh.
 *    - reduction_groups factors are dropped too, but for a different
 *      reason: a producing kernel's own local contraction is assumed to
 *      have already reduced over them before the value reached this
 *      collective. This is an assumption, not something this pass verifies
 *      structurally.
 *    - mapping_lhs/mapping_rhs are different again: a DeviceLocalAxis
 *      factor there is never dropped, only REPLACED in place by a
 *      same-extent factor for the tensor dimension it belongs to. Dropping
 *      it here would erase exactly the information a downstream lowering
 *      needs to tell a scatter or gather (a tensor dimension trading places
 *      with a mesh dimension) apart from an ordinary pass-through. See
 *      rebuildMappingFactorsForTarget's comment for the mechanics.
 *    - output_type/async_handle are grown, like a cast's or kernel's own
 *      result. Doing that per dimension requires knowing which output
 *      dimension a given factor belongs to -- input_mesh/output_mesh never
 *      need that association; see computeMappingRhsDeviceLocalGrowth's
 *      comment for how the mapping recovers it. A collective is assumed to
 *      be "bookended" (input_object directly produced by a
 *      DistributedCastGlobalToLocalOp) rather than chained (fed by
 *      DistributedAwait on another collective); see the comment above the
 *      mapping loop in inlineDeviceLocalAxesInCollective for what this pass
 *      does and doesn't verify about that.
 *
 * Known open gap this pass doesn't address:
 * CanonicalizeShardedFactorOrderPass's own doc comment describes an
 * as-yet-unimplemented "reconciling DistributedCollectiveOp" needed to bridge a
 * cast whose partitioning_axes it canonically reordered with a real collective
 * that still expects the original (pre-reorder) view. If that reconciling
 * mechanism is ever built, revisit whether this pass's mapping rebuild still
 * composes correctly with it.
 *
 * See the comment on each function below for the mechanics and why each
 * case is safe.
 */

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_INLINEDEVICELOCALAXESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Splits a factor-group's raw factors into (kept, deviceLocalExtent): kept is
// every non-DeviceLocalAxisType-provenance factor (original order preserved),
// deviceLocalExtent is the product of extents of every DeviceLocalAxisType
// factor found (1 if none).
static FailureOr<
    std::pair<SmallVector<TypedValue<axis::AxisFactorType>>, int64_t>>
splitOutDeviceLocalFactors(TypedValue<axis::FactorGroupType> factorGroup) {
  auto factors = axis::getProductProvenanceFactors(factorGroup);
  if (failed(factors)) {
    return failure();
  }
  SmallVector<TypedValue<axis::AxisFactorType>> kept;
  int64_t deviceLocalExtent = 1;
  for (auto factor : *factors) {
    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (failed(provenance)) {
      return failure();
    }
    if (isa<DeviceLocalAxisType>(provenance->getType())) {
      deviceLocalExtent *= axis::getFactorExtent(factor);
    } else {
      kept.push_back(factor);
    }
  }
  return std::make_pair(kept, deviceLocalExtent);
}

// Same as splitOutDeviceLocalFactors, but for a kernel-style dimension: a
// list of slot indices into `partitioningAxes`. Only the combined
// DeviceLocalAxis extent is needed -- kernel shardings/partitioning_axes are
// never rewritten (see the file doc comment).
static FailureOr<int64_t>
computeDeviceLocalGrowthFactor(ValueRange partitioningAxes,
                               ArrayRef<int64_t> axisIndices) {
  int64_t growth = 1;
  for (int64_t idx : axisIndices) {
    if (idx < 0 || idx >= static_cast<int64_t>(partitioningAxes.size())) {
      return failure();
    }
    auto factorGroup =
        cast<TypedValue<axis::FactorGroupType>>(partitioningAxes[idx]);
    auto factors = axis::getProductProvenanceFactors(factorGroup);
    if (failed(factors)) {
      return failure();
    }
    for (auto factor : *factors) {
      auto provenance = axis::getFactorProvenanceAxis(factor);
      if (failed(provenance)) {
        return failure();
      }
      if (isa<DeviceLocalAxisType>(provenance->getType())) {
        growth *= axis::getFactorExtent(factor);
      }
    }
  }
  return growth;
}

// Scans one mapping_rhs entry (a flat factor-group product covering every
// output tensor dimension) and returns, per dimension index, the product of
// DeviceLocalAxis factor extents found for that dimension: how much
// output_type needs to grow there.
//
// Mesh operands (input_mesh/output_mesh) never need to know which dimension
// a factor belongs to; they're flat totals, stripped or compared as a
// whole. The mapping does need that association, and recovers it
// positionally:
//  - MaterializeDistributedCollectives.cpp's toLocallyTypedAxisProduct
//    builds each mapping_lhs/mapping_rhs entry dimension by dimension, in
//    rank order: each dimension's own mesh factors, followed by exactly one
//    ShapeAxisType factor anchoring it (from axis.getaxis on that
//    dimension).
//  - So scanning front to back, everything since the last anchor belongs to
//    the dimension the next anchor names.
//  - That positional bookkeeping survives the search's factor rebasing.
//    axis::replaceAxisFactors (how SearchStrategies.cpp turns a
//    LogicalMeshAxis factor into physical/device-local ones) splices its
//    replacement in at the same operand position; it never appends or
//    reorders. The anchors themselves are never rebased (they're ShapeAxis,
//    not LogicalMeshAxis), so they stay exactly where construction put
//    them.
static FailureOr<llvm::DenseMap<unsigned, int64_t>>
computeMappingRhsDeviceLocalGrowth(TypedValue<axis::FactorGroupType> group) {
  auto factors = axis::getProductProvenanceFactors(group);
  if (failed(factors)) {
    return failure();
  }
  llvm::DenseMap<unsigned, int64_t> growthPerDim;
  int64_t runDeviceLocalExtent = 1;
  for (auto factor : *factors) {
    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (failed(provenance)) {
      return failure();
    }
    if (isa<axis::ShapeAxisType>(provenance->getType())) {
      unsigned dim = axis::getAxisDimIndex(
          cast<TypedValue<axis::ShapeAxisType>>(*provenance));
      growthPerDim[dim] = runDeviceLocalExtent;
      runDeviceLocalExtent = 1;
    } else if (isa<DeviceLocalAxisType>(provenance->getType())) {
      runDeviceLocalExtent *= axis::getFactorExtent(factor);
    }
  }
  if (runDeviceLocalExtent != 1) {
    // DeviceLocalAxis factor(s) after the last anchor: can't tell which
    // dimension they belong to.
    return failure();
  }
  return growthPerDim;
}

// Computes collectiveOp's grown output_type, using the growth map derived
// from `mappingRhs` (see computeMappingRhsDeviceLocalGrowth above). Returns
// `currentType` unchanged if no growth is needed. Pure computation -- does
// not mutate anything.
static FailureOr<RankedTensorType>
computeGrownOutputType(RankedTensorType currentType, ValueRange mappingRhs) {
  SmallVector<int64_t> newShape(currentType.getShape());
  bool changed = false;
  for (Value rhs : mappingRhs) {
    auto typedRhs = cast<TypedValue<axis::FactorGroupType>>(rhs);
    auto growthPerDim = computeMappingRhsDeviceLocalGrowth(typedRhs);
    if (failed(growthPerDim)) {
      return failure();
    }
    for (auto &[dim, growth] : *growthPerDim) {
      if (growth == 1) {
        continue;
      }
      if (dim >= newShape.size()) {
        return failure();
      }
      newShape[dim] *= growth;
      changed = true;
    }
  }
  if (!changed) {
    return currentType;
  }
  return RankedTensorType::get(newShape, currentType.getElementType());
}

// Sets collectiveOp's own output_type attribute and async_handle result type
// to `newType` (a no-op if unchanged), and propagates the change to any
// DistributedAwait uses of the async_handle. Those won't otherwise pick up
// the change: DistributedAwait::inferReturnTypes only runs once, at
// construction.
static void applyCollectiveOutputType(DistributedCollectiveOp collectiveOp,
                                      RankedTensorType newType) {
  if (newType == collectiveOp.getOutputType()) {
    return;
  }
  collectiveOp->setAttr("output_type", TypeAttr::get(newType));
  collectiveOp.getAsyncHandle().setType(
      AsynchHandleType::get(collectiveOp.getContext(), newType));
  for (OpOperand &use : collectiveOp.getAsyncHandle().getUses()) {
    if (auto awaitOp = dyn_cast<DistributedAwait>(use.getOwner())) {
      awaitOp.getValue().setType(newType);
    }
  }
}

// Where a single factor's same-extent replacement should go: which tensor
// dimension, and what stride within it.
struct DeviceLocalReplacementSite {
  unsigned dim;
  int64_t stride;
};

// A plan for rewriting one mapping side: which factor positions (indices
// into the factor list) need to become a ShapeAxisType factor instead of a
// DeviceLocalAxisType one, and each affected dimension's combined
// DeviceLocalAxis extent (needed to place that dimension's own original
// anchor above them).
struct MappingReplacementPlan {
  llvm::DenseMap<size_t, DeviceLocalReplacementSite> perFactor;
  llvm::DenseMap<unsigned, int64_t> combinedExtentPerDim;
};

// Figures out where each DeviceLocalAxis factor in one mapping side's flat
// factor list belongs: which tensor dimension (named by that dimension's
// own trailing ShapeAxisType anchor -- see computeMappingRhsDeviceLocalGrowth's
// comment for why this positional scan is valid) and what stride its
// same-extent replacement should have there. DeviceLocalAxis factors within
// one dimension are placed minor-most, in their existing relative order
// (this pass's own contiguous/minor-most precondition) -- deliberately not
// merged into one combined factor; that consolidation, if ever wanted, is a
// separate canonicalization pass's job, not this one's.
static FailureOr<MappingReplacementPlan> planDeviceLocalReplacements(
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> factors) {
  MappingReplacementPlan plan;
  SmallVector<size_t> pendingIndices;
  int64_t pendingStride = 1;
  for (auto [idx, factor] : llvm::enumerate(factors)) {
    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (failed(provenance)) {
      return failure();
    }
    if (isa<DeviceLocalAxisType>(provenance->getType())) {
      pendingIndices.push_back(idx);
      continue;
    }
    auto shapeAxisType = dyn_cast<axis::ShapeAxisType>(provenance->getType());
    if (!shapeAxisType) {
      continue;
    }
    unsigned dim = shapeAxisType.getAxisIndex();
    for (size_t pendingIdx : pendingIndices) {
      plan.perFactor[pendingIdx] = {dim, pendingStride};
      pendingStride *= axis::getFactorExtent(factors[pendingIdx]);
    }
    if (!pendingIndices.empty()) {
      plan.combinedExtentPerDim[dim] = pendingStride;
    }
    pendingIndices.clear();
    pendingStride = 1;
  }
  if (!pendingIndices.empty()) {
    // DeviceLocalAxis factor(s) with no anchor to name their dimension.
    return failure();
  }
  return plan;
}

// Rebuilds one mapping side's flat factor-group product against
// `targetType` (input_object's own type for mapping_lhs, or the
// collective's newly-grown output_type for mapping_rhs): a pointwise
// rewrite using the plan above -- each factor either passes through
// unchanged or is swapped for a same-extent factor at the same position,
// never reordered.
//
// A DeviceLocalAxis factor is never dropped or merged: it's REPLACED with a
// same-extent ShapeAxisType factor. The dimension's original anchor keeps
// its own original extent -- never touched -- but its stride shifts to sit
// above the replacement factors. Relabeling like this, rather than merging
// DeviceLocalAxis's extent into the anchor, is what lets a downstream
// lowering still see a genuine scatter or gather (a tensor dimension
// trading places with a mesh dimension) instead of silently collapsing it.
// It also means this side's own total extent never changes, which keeps
// axis.map's own lhs/rhs extent invariant satisfied automatically.
//
// Returns the (possibly unchanged) rebuilt value, whether anything actually
// changed, and whether any DeviceLocalAxis factor was found on this side at
// all (the caller uses this to decide whether growth was expected here).
static FailureOr<std::tuple<Value, bool, bool>>
rebuildMappingFactorsForTarget(TypedValue<axis::FactorGroupType> group,
                               RankedTensorType targetType, OpBuilder &builder,
                               Location loc) {
  auto factors = axis::getProductProvenanceFactors(group);
  if (failed(factors)) {
    return failure();
  }
  auto plan = planDeviceLocalReplacements(*factors);
  if (failed(plan)) {
    return failure();
  }

  bool hadDeviceLocal = !plan->perFactor.empty();
  bool changed = false;
  SmallVector<Value> rebuilt;
  llvm::DenseMap<unsigned, Value> canonicalAxisForDim;
  auto getCanonicalAxis = [&](unsigned dim) -> Value {
    auto [it, inserted] = canonicalAxisForDim.try_emplace(dim);
    if (inserted) {
      it->second =
          builder.create<axis::AxisGetAxisOp>(loc, targetType, dim).getAxis();
    }
    return it->second;
  };

  for (auto [idx, factor] : llvm::enumerate(*factors)) {
    auto siteIt = plan->perFactor.find(idx);
    if (siteIt != plan->perFactor.end()) {
      // A DeviceLocalAxis factor with a known replacement site.
      if (!targetType ||
          siteIt->second.dim >= static_cast<unsigned>(targetType.getRank())) {
        return failure();
      }
      changed = true;
      rebuilt.push_back(builder
                            .create<axis::AxisFactorOp>(
                                loc, getCanonicalAxis(siteIt->second.dim),
                                axis::getFactorExtent(factor),
                                siteIt->second.stride)
                            .getResult());
      continue;
    }

    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (failed(provenance)) {
      return failure();
    }
    auto shapeAxisType = dyn_cast<axis::ShapeAxisType>(provenance->getType());
    if (!shapeAxisType) {
      // An ordinary mesh factor: unaffected by growth.
      rebuilt.push_back(factor);
      continue;
    }

    // This dimension's own anchor.
    unsigned dim = shapeAxisType.getAxisIndex();
    auto combinedExtentIt = plan->combinedExtentPerDim.find(dim);
    if (combinedExtentIt == plan->combinedExtentPerDim.end() &&
        (!targetType || shapeAxisType.getShapeType() == targetType)) {
      // Nothing about this dimension changed.
      rebuilt.push_back(factor);
      continue;
    }
    int64_t oldAnchorExtent = axis::getFactorExtent(factor);
    int64_t deviceLocalTotalExtent =
        combinedExtentIt == plan->combinedExtentPerDim.end()
            ? 1
            : combinedExtentIt->second;
    if (!targetType || dim >= static_cast<unsigned>(targetType.getRank()) ||
        targetType.getDimSize(dim) !=
            oldAnchorExtent * deviceLocalTotalExtent) {
      // Either no real type to rebuild against, or targetType's dimension
      // doesn't actually reflect the growth these DeviceLocalAxis factors
      // imply -- can't safely resolve a replacement.
      return failure();
    }
    changed = true;
    rebuilt.push_back(
        builder
            .create<axis::AxisFactorOp>(loc, getCanonicalAxis(dim),
                                        oldAnchorExtent, deviceLocalTotalExtent)
            .getResult());
  }

  Value result = changed
                     ? Value(axis::viewFactorsAsProduct(rebuilt, builder, loc))
                     : Value(group);
  return std::make_tuple(result, changed, hadDeviceLocal);
}

// Grows `localSide`'s type in place (dimension-by-dimension, by each
// dimension's DeviceLocalAxis factors' extent) and rewrites
// `castOp`'s partitioning_axes to drop those factors. `localSide` must be a
// value this op itself produces (DistributedCastGlobalToLocalOp's own
// result); for DistributedCastLocalToGlobalOp, whose local side is its own
// *operand*, pass std::nullopt and only partitioning_axes is rewritten.
template <typename CastOpTy>
static bool inlineDeviceLocalAxesInCast(CastOpTy castOp,
                                        bool localSideIsOwnResult) {
  bool sawUnsupported = false;
  ValueRange partitioningAxes = castOp.getPartitioningAxes();
  SmallVector<Value> newPartitioningAxes(partitioningAxes.begin(),
                                         partitioningAxes.end());
  OpBuilder builder(castOp);

  Value localSide = localSideIsOwnResult ? Value(castOp.getOutput())
                                         : Value(castOp.getInput());
  auto localType = dyn_cast<RankedTensorType>(localSide.getType());
  if (!localType) {
    return false;
  }
  SmallVector<int64_t> newLocalShape(localType.getShape());
  bool changed = false;

  for (auto [dim, factorGroupValue] : llvm::enumerate(partitioningAxes)) {
    auto factorGroup =
        cast<TypedValue<axis::FactorGroupType>>(factorGroupValue);
    auto split = splitOutDeviceLocalFactors(factorGroup);
    if (failed(split)) {
      castOp->emitRemark() << "inline-device-local-axes: partitioning_axes["
                           << dim << "] couldn't be resolved to raw factors";
      sawUnsupported = true;
      continue;
    }
    auto &[kept, deviceLocalExtent] = *split;
    if (deviceLocalExtent == 1) {
      continue;
    }
    newLocalShape[dim] *= deviceLocalExtent;
    newPartitioningAxes[dim] =
        axis::viewFactorsAsProduct(kept, builder, castOp.getLoc());
    changed = true;
  }

  if (!changed) {
    return sawUnsupported;
  }

  castOp.getPartitioningAxesMutable().assign(newPartitioningAxes);
  if (localSideIsOwnResult) {
    localSide.setType(
        RankedTensorType::get(newLocalShape, localType.getElementType()));
  }
  return sawUnsupported;
}

// Result of trying to grow one value's shape against one sharding entry's
// per-dimension partitioning-axis slots (see computeGrownBoundaryShape).
struct GrownBoundaryShape {
  // Set only if at least one dimension actually grew.
  std::optional<SmallVector<int64_t>> newShape;
  // Dimensions whose partitioning-axis slot(s) couldn't be resolved to raw
  // factors; growth for these was silently skipped, not assumed to be 1.
  SmallVector<unsigned> unresolvedDims;
};

// Computes rankedType's shape after growing each dimension by its own
// DeviceLocalAxis factors' combined extent, per `dimAxesList` (one
// DenseI64ArrayAttr of partitioning-axis slot indices per dimension, as
// stored on an IndexedTensorShardingAttr) against `partitioningAxes`. Shared
// by both a kernel's own result boundary and its operand-side
// UnrealizedConversionCastOp boundary, which otherwise repeat the identical
// per-dimension growth loop.
static GrownBoundaryShape
computeGrownBoundaryShape(RankedTensorType rankedType,
                          ArrayRef<DenseI64ArrayAttr> dimAxesList,
                          ValueRange partitioningAxes) {
  GrownBoundaryShape result;
  if (static_cast<int64_t>(dimAxesList.size()) != rankedType.getRank()) {
    return result;
  }
  SmallVector<int64_t> newShape(rankedType.getShape());
  bool changed = false;
  for (auto [dim, dimAxes] : llvm::enumerate(dimAxesList)) {
    auto growth =
        computeDeviceLocalGrowthFactor(partitioningAxes, dimAxes.asArrayRef());
    if (failed(growth)) {
      result.unresolvedDims.push_back(dim);
      continue;
    }
    if (*growth == 1) {
      continue;
    }
    newShape[dim] *= *growth;
    changed = true;
  }
  if (changed) {
    result.newShape = std::move(newShape);
  }
  return result;
}

// Grows every dimension of `kernelOp`'s own external result types using
// output_shardings' DeviceLocalAxis factors. This is the result-side
// counterpart of growing a DistributedCastGlobalToLocalOp's own result.
static bool inlineDeviceLocalAxesInKernelResults(DistributedKernelOp kernelOp) {
  bool sawUnsupported = false;
  ArrayRef<IndexedTensorShardingAttr> outputShardings =
      kernelOp.getOutputShardings().getShardings();

  for (auto [resultIdx, result] : llvm::enumerate(kernelOp.getReturns())) {
    if (resultIdx >= outputShardings.size()) {
      break;
    }
    auto rankedType = dyn_cast<RankedTensorType>(result.getType());
    if (!rankedType) {
      continue;
    }
    auto grown = computeGrownBoundaryShape(
        rankedType, outputShardings[resultIdx].getDimPartitioningAxes(),
        kernelOp.getPartitioningAxes());
    for (unsigned dim : grown.unresolvedDims) {
      kernelOp.emitRemark()
          << "inline-device-local-axes: result " << resultIdx << "'s dimension "
          << dim
          << " references a partitioning-axis slot whose provenance "
             "couldn't be resolved";
      sawUnsupported = true;
    }
    if (grown.newShape) {
      result.setType(
          RankedTensorType::get(*grown.newShape, rankedType.getElementType()));
    }
  }
  return sawUnsupported;
}

// Grows a kernel operand's producer when that producer is a plain
// UnrealizedConversionCastOp: the placeholder ClusterDistributedKernels.cpp
// leaves at a kernel's own argument boundary whenever the chosen input
// value's type doesn't already match the kernel's expected (local) operand
// type.
//
// Unlike DistributedCastGlobalToLocalOp, this op has no partitioning_axes of
// its own, so growth is computed straight from the consuming kernel's own
// argument_shardings. Growing its result is always safe, since
// UnrealizedConversionCastOp has no verifier tying its input and output
// types together.
static bool growKernelOperandCasts(DistributedKernelOp kernelOp) {
  bool sawUnsupported = false;
  ArrayRef<IndexedTensorShardingAttr> argumentShardings =
      kernelOp.getArgumentShardings().getShardings();
  llvm::SmallPtrSet<Operation *, 4> processedCasts;

  for (auto [argIdx, operand] : llvm::enumerate(kernelOp.getArguments())) {
    if (argIdx >= argumentShardings.size()) {
      break;
    }
    auto castOp = operand.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1 ||
        castOp.getNumResults() != 1 || !processedCasts.insert(castOp).second) {
      continue;
    }
    auto rankedType = dyn_cast<RankedTensorType>(operand.getType());
    if (!rankedType) {
      continue;
    }
    auto grown = computeGrownBoundaryShape(
        rankedType, argumentShardings[argIdx].getDimPartitioningAxes(),
        kernelOp.getPartitioningAxes());
    for (unsigned dim : grown.unresolvedDims) {
      kernelOp.emitRemark()
          << "inline-device-local-axes: operand " << argIdx << "'s dimension "
          << dim
          << " references a partitioning-axis slot whose provenance "
             "couldn't be resolved";
      sawUnsupported = true;
    }
    if (grown.newShape) {
      castOp.getResult(0).setType(
          RankedTensorType::get(*grown.newShape, rankedType.getElementType()));
    }
  }
  return sawUnsupported;
}

// Debug/validation check: flags a kernel operand that needs DeviceLocalAxis
// growth per its own argument_shardings but isn't produced by something this
// pass already grows (a cast, a kernel, an UnrealizedConversionCastOp, or a
// DistributedAwait). Collectives are processed before this check runs, so an
// Await producer is always safe to accept here.
//
// Remark only, never a hard failure. SearchStrategies.cpp runs this pass
// internally to score every candidate during search, and collective-fed
// kernels (e.g. any all-reduce) are completely ordinary; failing here would
// make the search unable to find any usable candidate for such models.
static void checkKernelOperandGrowthConsistency(DistributedKernelOp kernelOp) {
  ArrayRef<IndexedTensorShardingAttr> argumentShardings =
      kernelOp.getArgumentShardings().getShardings();

  for (auto [argIdx, operand] : llvm::enumerate(kernelOp.getArguments())) {
    if (argIdx >= argumentShardings.size()) {
      break;
    }
    auto rankedType = dyn_cast<RankedTensorType>(operand.getType());
    if (!rankedType) {
      continue;
    }
    ArrayRef<DenseI64ArrayAttr> dimAxesList =
        argumentShardings[argIdx].getDimPartitioningAxes();
    if (static_cast<int64_t>(dimAxesList.size()) != rankedType.getRank()) {
      continue;
    }

    bool needsRealGrowth = false;
    for (DenseI64ArrayAttr dimAxes : dimAxesList) {
      auto growth = computeDeviceLocalGrowthFactor(
          kernelOp.getPartitioningAxes(), dimAxes.asArrayRef());
      if (succeeded(growth) && *growth > 1) {
        needsRealGrowth = true;
        break;
      }
    }
    if (!needsRealGrowth) {
      continue;
    }

    Operation *producer = operand.getDefiningOp();
    if (isa_and_nonnull<DistributedCastGlobalToLocalOp, DistributedKernelOp,
                        UnrealizedConversionCastOp, DistributedAwait>(
            producer)) {
      continue;
    }
    kernelOp.emitRemark()
        << "inline-device-local-axes: operand " << argIdx
        << " needs DeviceLocalAxis growth per its own argument_shardings, "
           "but isn't produced by a distributed.CastGlobalToLocal, "
           "distributed.DistributedKernel, builtin.unrealized_conversion_cast, "
           "or distributed.Await this pass could have already grown";
  }
}

// Strips DeviceLocalAxis factors from a collective's own input_mesh/
// output_mesh/reduction_groups/mapping bookkeeping, and grows its own
// output_type/async_handle result (see the file doc comment for the two
// different reasons factors get dropped from mesh/mapping vs.
// reduction_groups). input_object is never touched; it's expected to
// already be grown by whatever produced it.
static bool
inlineDeviceLocalAxesInCollective(DistributedCollectiveOp collectiveOp) {
  OpBuilder builder(collectiveOp);
  Location loc = collectiveOp.getLoc();

  auto stripGroup = [&](TypedValue<axis::FactorGroupType> group)
      -> FailureOr<std::pair<Value, bool>> {
    auto split = splitOutDeviceLocalFactors(group);
    if (failed(split)) {
      return failure();
    }
    auto &[kept, deviceLocalExtent] = *split;
    if (deviceLocalExtent == 1) {
      return std::make_pair(Value(group), false);
    }
    return std::make_pair(Value(axis::viewFactorsAsProduct(kept, builder, loc)),
                          true);
  };

  auto newInputMesh = stripGroup(collectiveOp.getInputMesh());
  auto newOutputMesh = stripGroup(collectiveOp.getOutputMesh());
  if (failed(newInputMesh) || failed(newOutputMesh)) {
    collectiveOp.emitRemark() << "inline-device-local-axes: input_mesh/"
                                 "output_mesh couldn't be resolved to raw "
                                 "factors";
    return true;
  }

  ValueRange oldReductionGroups = collectiveOp.getReductionGroups();
  MutableArrayRef<Region> oldBodies = collectiveOp.getReductionBodies();
  SmallVector<Value> newReductionGroups;
  SmallVector<Region *> keptBodies;
  bool reductionGroupRemoved = false;
  bool reductionContentChanged = false;
  bool reductionAssumedAlreadyLocal = false;
  for (size_t i = 0; i < oldReductionGroups.size(); ++i) {
    auto typedGroup =
        cast<TypedValue<axis::FactorGroupType>>(oldReductionGroups[i]);
    auto split = splitOutDeviceLocalFactors(typedGroup);
    if (failed(split)) {
      collectiveOp.emitRemark() << "inline-device-local-axes: reduction "
                                   "group "
                                << i << " couldn't be resolved to raw factors";
      return true;
    }
    auto &[kept, deviceLocalExtent] = *split;
    if (kept.empty()) {
      // Every factor in this group is DeviceLocalAxis-provenance. Dropping
      // it assumes the producer's own local contraction already reduced
      // over it before the value reached this collective; the producer-kind
      // check below is the only verification of that.
      reductionGroupRemoved = true;
      reductionAssumedAlreadyLocal = true;
      continue;
    }
    if (deviceLocalExtent != 1) {
      newReductionGroups.push_back(
          axis::viewFactorsAsProduct(kept, builder, loc));
      reductionContentChanged = true;
    } else {
      newReductionGroups.push_back(oldReductionGroups[i]);
    }
    keptBodies.push_back(&oldBodies[i]);
  }

  auto mappingOp = collectiveOp.getMapping().getDefiningOp<axis::AxisMapOp>();
  if (!mappingOp) {
    collectiveOp.emitRemark()
        << "inline-device-local-axes: mapping isn't produced by axis.map";
    return true;
  }
  ValueRange oldMappingLhs = mappingOp.getMappingLhs();
  ValueRange oldMappingRhs = mappingOp.getMappingRhs();

  // Computed up front (from the *old*, pre-strip mapping_rhs) because
  // rebuilding mapping_rhs's own factors below needs to know the target type
  // its ShapeAxis anchors should be rebuilt against.
  auto currentOutputType =
      dyn_cast<RankedTensorType>(collectiveOp.getOutputType());
  RankedTensorType newOutputType = currentOutputType;
  if (currentOutputType) {
    auto grown = computeGrownOutputType(currentOutputType, oldMappingRhs);
    if (failed(grown)) {
      collectiveOp.emitRemark() << "inline-device-local-axes: mapping_rhs "
                                   "couldn't be resolved for output_type "
                                   "growth";
      return true;
    }
    newOutputType = *grown;
  }

  // Used to rebuild mapping_lhs's own ShapeAxis anchors if input_object's
  // type has already grown relative to what they were built against.
  // input_object itself is never grown here.
  //
  // Assumption: a collective is "bookended" -- input_object is directly
  // produced by a DistributedCastGlobalToLocalOp, matching how
  // MaterializeDistributedCollectives.cpp normally builds one. This pass
  // doesn't independently verify that for a *chained* collective
  // (input_object fed by DistributedAwait on another collective instead);
  // the only thing that actually catches a violation is the numeric
  // dimension-size check in rebuildMappingFactorsForTarget, which fails
  // (remark, non-fatal) rather than silently mis-rewriting. See the file doc
  // comment's note on CanonicalizeShardedFactorOrderPass's unimplemented
  // reconciling collective for a related, still-open gap.
  auto inputObjectType =
      dyn_cast<RankedTensorType>(collectiveOp.getInputObject().getType());

  SmallVector<Value> newMappingLhs, newMappingRhs;
  bool mappingChanged = false;
  bool inputGrowthExpected = false;
  for (size_t i = 0; i < oldMappingLhs.size(); ++i) {
    auto typedLhs = cast<TypedValue<axis::FactorGroupType>>(oldMappingLhs[i]);
    auto typedRhs = cast<TypedValue<axis::FactorGroupType>>(oldMappingRhs[i]);
    auto lhsResult =
        rebuildMappingFactorsForTarget(typedLhs, inputObjectType, builder, loc);
    auto rhsResult =
        rebuildMappingFactorsForTarget(typedRhs, newOutputType, builder, loc);
    if (failed(lhsResult) || failed(rhsResult)) {
      collectiveOp.emitRemark() << "inline-device-local-axes: mapping pair "
                                << i << " couldn't be resolved to raw factors";
      return true;
    }
    auto &[lhsValue, lhsChanged, lhsHadDeviceLocal] = *lhsResult;
    auto &[rhsValue, rhsChanged, rhsHadDeviceLocal] = *rhsResult;
    if (lhsHadDeviceLocal) {
      inputGrowthExpected = true;
    }
    newMappingLhs.push_back(lhsValue);
    newMappingRhs.push_back(rhsValue);
    if (lhsChanged || rhsChanged) {
      mappingChanged = true;
    }
  }

  // Remark-only, same style as checkKernelOperandGrowthConsistency:
  // input_object is never grown here, so if mapping_lhs implies real growth,
  // its producer had better already be something that grows its own result.
  if (inputGrowthExpected) {
    Operation *producer = collectiveOp.getInputObject().getDefiningOp();
    if (!isa_and_nonnull<DistributedCastGlobalToLocalOp, DistributedKernelOp,
                         DistributedAwait>(producer)) {
      collectiveOp.emitRemark()
          << "inline-device-local-axes: input_object needs DeviceLocalAxis "
             "growth per its mapping, but isn't produced by a "
             "distributed.CastGlobalToLocal, distributed.DistributedKernel, "
             "or distributed.Await this pass could have already grown";
    } else if (isa<DistributedAwait>(producer)) {
      // Not an untrusted producer (this pass already grows Awaits), but
      // still worth surfacing: this collective is chained, not bookended
      // (see the assumption above), and this pass doesn't independently
      // verify beyond the numeric check in rebuildMappingFactorsForTarget.
      collectiveOp.emitRemark()
          << "inline-device-local-axes: input_object is a DistributedAwait "
             "on another collective (a chained, not bookended, collective); "
             "this pass's mapping_lhs rebuild here relies on an assumption "
             "it does not independently verify beyond a dimension-size "
             "check";
    }
  }

  // Remark-only: a dropped reduction group assumed some producer's own
  // local contraction already reduced over it (see the loop above). That
  // assumption only makes sense if a kernel is actually in the producer
  // chain, so flag the case where it plainly isn't.
  if (reductionAssumedAlreadyLocal) {
    Operation *producer = collectiveOp.getInputObject().getDefiningOp();
    if (!isa_and_nonnull<DistributedCastGlobalToLocalOp, DistributedKernelOp,
                         DistributedAwait>(producer)) {
      collectiveOp.emitRemark()
          << "inline-device-local-axes: dropped a reduction group assuming "
             "it was already reduced locally, but input_object isn't "
             "produced by a distributed.CastGlobalToLocal, "
             "distributed.DistributedKernel, or distributed.Await, so that "
             "assumption may not hold here";
    }
  }

  bool meshChanged = newInputMesh->second || newOutputMesh->second;
  if (!meshChanged && !reductionGroupRemoved && !reductionContentChanged &&
      !mappingChanged) {
    return false;
  }

  Value finalMapping = collectiveOp.getMapping();
  if (mappingChanged) {
    auto newMapOp = builder.create<axis::AxisMapOp>(
        loc, axis::AxisMapType::get(builder.getContext()), newMappingLhs,
        newMappingRhs);
    finalMapping = newMapOp.getMap();
  }

  if (!reductionGroupRemoved) {
    // reduction_bodies' region count is unaffected -- safe to mutate this
    // op's operands in place.
    collectiveOp.getInputMeshMutable().assign(newInputMesh->first);
    collectiveOp.getOutputMeshMutable().assign(newOutputMesh->first);
    collectiveOp.getReductionGroupsMutable().assign(newReductionGroups);
    collectiveOp.getMappingMutable().assign(finalMapping);
    if (currentOutputType) {
      applyCollectiveOutputType(collectiveOp, newOutputType);
    }
    return false;
  }

  // A whole reduction group vanished: reduction_bodies' region count must
  // shrink to match, which requires rebuilding the op (regions can't be
  // removed from an existing op in place).
  Type finalOutputType =
      currentOutputType ? Type(newOutputType) : collectiveOp.getOutputType();
  OperationState state(loc, DistributedCollectiveOp::getOperationName());
  state.addOperands(collectiveOp.getInputObject());
  state.addOperands(newInputMesh->first);
  state.addOperands(newOutputMesh->first);
  state.addOperands(newReductionGroups);
  state.addOperands(finalMapping);
  state.addTypes(
      AsynchHandleType::get(collectiveOp.getContext(), finalOutputType));
  state.addAttribute("output_type", TypeAttr::get(finalOutputType));
  for (size_t i = 0; i < newReductionGroups.size(); ++i) {
    state.addRegion();
  }
  auto newOp = cast<DistributedCollectiveOp>(builder.create(state));
  for (size_t newIdx = 0; newIdx < keptBodies.size(); ++newIdx) {
    newOp.getReductionBodies()[newIdx].takeBody(*keptBodies[newIdx]);
  }
  for (OpOperand &use : collectiveOp.getAsyncHandle().getUses()) {
    if (auto awaitOp = dyn_cast<DistributedAwait>(use.getOwner())) {
      awaitOp.getValue().setType(finalOutputType);
    }
  }
  collectiveOp.getAsyncHandle().replaceAllUsesWith(newOp.getAsyncHandle());
  collectiveOp.erase();
  return false;
}

} // namespace

struct InlineDeviceLocalAxesPass
    : public impl::InlineDeviceLocalAxesPassBase<InlineDeviceLocalAxesPass> {
  using InlineDeviceLocalAxesPassBase::InlineDeviceLocalAxesPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool sawUnsupported = false;

    moduleOp.walk([&](Operation *op) {
      if (auto g2l = dyn_cast<DistributedCastGlobalToLocalOp>(op)) {
        if (inlineDeviceLocalAxesInCast(g2l, /*localSideIsOwnResult=*/true)) {
          sawUnsupported = true;
        }
      } else if (auto l2g = dyn_cast<DistributedCastLocalToGlobalOp>(op)) {
        if (inlineDeviceLocalAxesInCast(l2g, /*localSideIsOwnResult=*/false)) {
          sawUnsupported = true;
        }
      } else if (auto kernelOp = dyn_cast<DistributedKernelOp>(op)) {
        if (inlineDeviceLocalAxesInKernelResults(kernelOp)) {
          sawUnsupported = true;
        }
        if (growKernelOperandCasts(kernelOp)) {
          sawUnsupported = true;
        }
      }
    });

    // Collected up front rather than handled inline above: a collective may
    // need to be entirely rebuilt/erased (see
    // inlineDeviceLocalAxesInCollective's own comment on region-count
    // changes), which isn't safe to do to the very op a walk() is currently
    // visiting.
    SmallVector<DistributedCollectiveOp> collectiveOps;
    moduleOp.walk([&](DistributedCollectiveOp collectiveOp) {
      collectiveOps.push_back(collectiveOp);
    });
    for (DistributedCollectiveOp collectiveOp : collectiveOps) {
      if (inlineDeviceLocalAxesInCollective(collectiveOp)) {
        sawUnsupported = true;
      }
    }

    // Consistency check runs only after every cast/kernel result has already
    // been grown, so every kernel operand this pass could have grown "for
    // free" has actually done so by now. Remark-only (see its own comment),
    // so it does not contribute to sawUnsupported.
    moduleOp.walk([&](DistributedKernelOp kernelOp) {
      checkKernelOperandGrowthConsistency(kernelOp);
    });

    if (sawUnsupported) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::enzyme::distributed
