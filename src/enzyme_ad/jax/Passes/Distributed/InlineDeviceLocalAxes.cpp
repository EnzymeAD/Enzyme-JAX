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
 *    are never grown directly. Every consumer of a value needing growth
 *    trusts that value's own producer to have already grown it, and
 *    identifies a producer able to do so by whether it implements
 *    PartitioningAnchorOpInterface (see Dialect/Distributed/Interfaces.td) --
 *    the op kind itself never matters, only whether it owns an authoritative
 *    partitioning binding for its own result.
 *
 * Cases handled, one per op kind:
 *  - DistributedCastGlobalToLocalOp: grows its own result (the local side)
 *    per dimension, and drops those factors from partitioning_axes.
 *  - DistributedCastLocalToGlobalOp: symmetric, but its local side is an
 *    operand (someone else's result), so only partitioning_axes is
 *    rewritten.
 *  - AnchorPartitioningOp: AllTypesMatch (no local side to grow), so only
 *    partitioning_axes' own content is stripped, keeping it in sync with
 *    whatever it feeds.
 *  - DistributedKernelOp: grows its own external result types; its own
 *    operands are left untouched, per the propagation assumption above.
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
 *      comment for how the mapping recovers it. Every collective is
 *      "bookended" by a PartitioningAnchorOpInterface producer (a real cast,
 *      an anchor, or a kernel boundary), so this pass never has to handle an
 *      unanchored input_object.
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

// Where a DeviceLocalAxis sits within one specific binding: which tensor
// dimension, and what stride among its (contiguous, minor-most -- this
// pass's own precondition) sibling DeviceLocalAxis factors on that same
// dimension. Stride is computed here rather than read off the factor's own
// axis::getFactorStride, which is relative to its declaring axis, not to
// this position.
struct AxisBinding {
  unsigned dim;
  int64_t stride;
};

// Maps each DeviceLocalAxis (keyed by its declaring op's result -- see
// splitOutDeviceLocalFactors) to where it sits in one specific binding.
// Never merge two of these: the same axis can legitimately sit on a
// different dimension of some other, unrelated tensor (e.g. a value and its
// own transpose), so a binding is only meaningful for the one value it was
// resolved from.
using AxisToDimensionMap = llvm::DenseMap<Value, AxisBinding>;

// Converts one getBindingInfoForOperand/Result result (per dimension, the
// factor-group values contributing to it) into an AxisToDimensionMap.
static AxisToDimensionMap
axisToDimensionMapFromBinding(ArrayRef<SmallVector<Value>> binding) {
  AxisToDimensionMap axisToDim;
  for (auto [dim, factorGroups] : llvm::enumerate(binding)) {
    // Only DeviceLocalAxis factors advance the running stride: a mesh
    // factor sharing the slot doesn't occupy any of this dimension's
    // device-local sub-range.
    int64_t stride = 1;
    for (Value factorGroupValue : factorGroups) {
      auto factorGroup =
          dyn_cast<TypedValue<axis::FactorGroupType>>(factorGroupValue);
      if (!factorGroup) {
        continue;
      }
      auto factors = axis::getProductProvenanceFactors(factorGroup);
      if (failed(factors)) {
        continue;
      }
      for (auto factor : *factors) {
        auto provenance = axis::getFactorProvenanceAxis(factor);
        if (failed(provenance) ||
            !isa<DeviceLocalAxisType>(provenance->getType())) {
          continue;
        }
        axisToDim.try_emplace(*provenance,
                              AxisBinding{static_cast<unsigned>(dim), stride});
        stride *= axis::getFactorExtent(factor);
      }
    }
  }
  return axisToDim;
}

// Resolves an operand's binding via its producer's own
// PartitioningAnchorOpInterface implementation -- one lookup, no walking:
// the interface is the sole authority on what a value's current binding is.
// Must run before anything in this pass mutates/strips a slot's own
// content, since that binding is what's being read here.
static AxisToDimensionMap resolveOperandAxisBindings(Value value) {
  if (!isa<RankedTensorType>(value.getType())) {
    return AxisToDimensionMap();
  }
  auto anchorOp =
      dyn_cast_or_null<PartitioningAnchorOpInterface>(value.getDefiningOp());
  if (!anchorOp) {
    return AxisToDimensionMap();
  }
  auto binding = anchorOp.getBindingInfoForResult(cast<OpResult>(value));
  if (failed(binding)) {
    return AxisToDimensionMap();
  }
  return axisToDimensionMapFromBinding(*binding);
}

// Resolves a result's binding via any one consumer implementing
// PartitioningAnchorOpInterface -- every such consumer must agree (this
// dialect's own consistency invariant, maintained elsewhere), so the first
// one found is as good as any other.
static AxisToDimensionMap resolveResultAxisBindings(Value value) {
  if (!isa<RankedTensorType>(value.getType())) {
    return AxisToDimensionMap();
  }
  for (OpOperand &use : value.getUses()) {
    auto anchorOp = dyn_cast<PartitioningAnchorOpInterface>(use.getOwner());
    if (!anchorOp) {
      continue;
    }
    auto binding = anchorOp.getBindingInfoForOperand(use);
    if (failed(binding)) {
      continue;
    }
    return axisToDimensionMapFromBinding(*binding);
  }
  return AxisToDimensionMap();
}

// Scans one mapping_rhs entry (a flat factor-group product covering every
// output tensor dimension) and returns, per dimension index, the product of
// DeviceLocalAxis factor extents found for that dimension: how much
// output_type needs to grow there.
//
// Mesh operands (input_mesh/output_mesh) never need to know which dimension
// a factor belongs to; they're flat totals, stripped or compared as a
// whole. The mapping does need that association, resolved two ways:
//  - Positionally: MaterializeDistributedCollectives.cpp builds each entry
//    dimension by dimension, each dimension's mesh factors followed by one
//    ShapeAxisType anchor naming it -- scanning front to back, everything
//    since the last anchor belongs to the dimension the next one names.
//  - Via `axisToDim` otherwise, for a mapping entry with no embedded anchor
//    at all (e.g. a genuine identity pass-through over these axes).
static FailureOr<llvm::DenseMap<unsigned, int64_t>>
computeMappingRhsDeviceLocalGrowth(TypedValue<axis::FactorGroupType> group,
                                   const AxisToDimensionMap &axisToDim) {
  auto factors = axis::getProductProvenanceFactors(group);
  if (failed(factors)) {
    return failure();
  }
  llvm::DenseMap<unsigned, int64_t> growthPerDim;
  auto accumulate = [&](unsigned dim, int64_t extent) {
    growthPerDim.try_emplace(dim, 1).first->second *= extent;
  };
  int64_t runDeviceLocalExtent = 1;
  for (auto factor : *factors) {
    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (failed(provenance)) {
      return failure();
    }
    if (isa<axis::ShapeAxisType>(provenance->getType())) {
      unsigned dim = axis::getAxisDimIndex(
          cast<TypedValue<axis::ShapeAxisType>>(*provenance));
      accumulate(dim, runDeviceLocalExtent);
      runDeviceLocalExtent = 1;
    } else if (isa<DeviceLocalAxisType>(provenance->getType())) {
      auto dimIt = axisToDim.find(*provenance);
      if (dimIt != axisToDim.end()) {
        accumulate(dimIt->second.dim, axis::getFactorExtent(factor));
      } else {
        runDeviceLocalExtent *= axis::getFactorExtent(factor);
      }
    }
  }
  if (runDeviceLocalExtent != 1) {
    // DeviceLocalAxis factor(s) after the last anchor, with no durable
    // binding in axisToDim either: genuinely can't tell which dimension
    // they belong to.
    return failure();
  }
  return growthPerDim;
}

// Computes collectiveOp's grown output_type, using the growth map derived
// from `mappingRhs` (see computeMappingRhsDeviceLocalGrowth above). Returns
// `currentType` unchanged if no growth is needed. Pure computation -- does
// not mutate anything.
static FailureOr<RankedTensorType>
computeGrownOutputType(RankedTensorType currentType, ValueRange mappingRhs,
                       const AxisToDimensionMap &axisToDim) {
  SmallVector<int64_t> newShape(currentType.getShape());
  bool changed = false;
  for (Value rhs : mappingRhs) {
    auto typedRhs = cast<TypedValue<axis::FactorGroupType>>(rhs);
    auto growthPerDim = computeMappingRhsDeviceLocalGrowth(typedRhs, axisToDim);
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

// AnchorPartitioningOp is a transparent, same-scope marker: it never owns
// its own growth decision, it only ever reflects whatever its real upstream
// producer (a Cast, a DistributedKernelOp, an Await -- something with its
// own authoritative growth logic elsewhere in this file) decides. Every
// growth site in this file must therefore mutate that authoritative
// producer's own value directly (never an operand/consumer value that merely
// happens to be fed by one), then forward-propagate here so any
// AnchorPartitioningOp between it and its consumers picks up the identical
// new type (AllTypesMatch requires the anchor's own result to match).
// Forward-only, recursing through chains of anchors: never walk backward
// into a producer's own type from a consumer's requirement -- that direction
// corrupts the producer's own, independently-checked invariant (e.g.
// DistributedKernelOp's own result/yield consistency, or a
// DistributedAwait's type racing its own collective's growth logic, which
// runs in a separate, later walk) with a size it never actually decided on.
static void propagateGrowthThroughAnchors(Value grownValue) {
  SmallVector<Value> worklist = {grownValue};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Type t = v.getType();
    for (OpOperand &use : v.getUses()) {
      auto anchor = dyn_cast<AnchorPartitioningOp>(use.getOwner());
      if (!anchor || use.getOperandNumber() != 0) {
        continue;
      }
      if (anchor.getOutput().getType() != t) {
        anchor.getOutput().setType(cast<TensorType>(t));
        worklist.push_back(anchor.getOutput());
      }
    }
  }
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
      propagateGrowthThroughAnchors(awaitOp.getValue());
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
// factor list belongs: which tensor dimension and what stride its
// same-extent replacement should have there. Prefers `axisToDim` per
// factor; falls back to the positional scan (named by that dimension's own
// trailing ShapeAxisType anchor in this same list, with DeviceLocalAxis
// factors placed minor-most in their existing relative order -- this
// pass's own contiguous/minor-most precondition) for whichever factors
// axisToDim doesn't know about.
static FailureOr<MappingReplacementPlan> planDeviceLocalReplacements(
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> factors,
    const AxisToDimensionMap &axisToDim) {
  MappingReplacementPlan plan;
  SmallVector<size_t> pendingIndices;
  int64_t pendingStride = 1;
  for (auto [idx, factor] : llvm::enumerate(factors)) {
    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (failed(provenance)) {
      return failure();
    }
    if (isa<DeviceLocalAxisType>(provenance->getType())) {
      auto dimIt = axisToDim.find(*provenance);
      if (dimIt != axisToDim.end()) {
        unsigned dim = dimIt->second.dim;
        plan.perFactor[idx] = {dim, dimIt->second.stride};
        // A real ShapeAxis anchor still present later in this list needs
        // this dimension's combined extent to shift its own stride -- the
        // same tally the positional pendingStride below feeds.
        auto [extentIt, inserted] =
            plan.combinedExtentPerDim.try_emplace(dim, 1);
        extentIt->second *= axis::getFactorExtent(factor);
        continue;
      }
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
    // DeviceLocalAxis factor(s) with no anchor to name their dimension,
    // positionally or via axisToDim.
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
static FailureOr<std::tuple<Value, bool, bool>> rebuildMappingFactorsForTarget(
    TypedValue<axis::FactorGroupType> group, RankedTensorType targetType,
    OpBuilder &builder, Location loc, const AxisToDimensionMap &axisToDim) {
  auto factors = axis::getProductProvenanceFactors(group);
  if (failed(factors)) {
    return failure();
  }
  auto plan = planDeviceLocalReplacements(*factors, axisToDim);
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
      // A DeviceLocalAxis factor with a known replacement site. Validate the
      // divisibility AxisFactorOp's own verifier requires (sourceExtent %
      // (stride * extent) == 0) before building it: AxisFactorOp's builder
      // fails type inference FATALLY (llvm::report_fatal_error, not a
      // LogicalResult) on a bad factor, which would otherwise crash the
      // whole compiler here instead of the graceful, non-fatal remark this
      // function's callers expect -- exactly the "chained, not bookended,
      // collective" scenario this file's top comment already documents as
      // possible (its input_object/output_type was never actually grown to
      // match what these DeviceLocalAxis factors imply).
      unsigned dim = siteIt->second.dim;
      int64_t extent = axis::getFactorExtent(factor);
      int64_t stride = siteIt->second.stride;
      if (!targetType || dim >= static_cast<unsigned>(targetType.getRank()) ||
          targetType.getDimSize(dim) % (stride * extent) != 0) {
        return failure();
      }
      changed = true;
      rebuilt.push_back(builder
                            .create<axis::AxisFactorOp>(
                                loc, getCanonicalAxis(dim), extent, stride)
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
  axis::ModuleScopeGuard moduleScope(builder);

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
    propagateGrowthThroughAnchors(localSide);
  }
  return sawUnsupported;
}

// AnchorPartitioningOp is AllTypesMatch (its input and output are always the
// identical type), so unlike a real Cast op there is no local side to grow
// here. But its partitioning_axes are, like a Cast's, the declared
// decomposition of its single (shared) input/output value -- and every other
// consumer of that same value independently strips a DeviceLocal factor from
// its own declared view once inlined (a kernel operand's slot, a collective's
// mapping). If the anchor sitting between them keeps declaring the old,
// un-stripped content forever, it disagrees with whatever it feeds: the
// anchor says "DeviceLocalAxis<8> still present here", its consumer says
// "nothing left to declare for this dimension" -- for the identical value.
// Pure content strip, mirroring the kernel-slot treatment: no tensor value
// or type ever changes here.
static bool inlineDeviceLocalAxesInAnchor(AnchorPartitioningOp anchorOp) {
  bool sawUnsupported = false;
  ValueRange partitioningAxes = anchorOp.getPartitioningAxes();
  SmallVector<Value> newPartitioningAxes(partitioningAxes.begin(),
                                         partitioningAxes.end());
  OpBuilder builder(anchorOp);
  axis::ModuleScopeGuard moduleScope(builder);
  bool changed = false;

  for (auto [dim, factorGroupValue] : llvm::enumerate(partitioningAxes)) {
    auto factorGroup =
        cast<TypedValue<axis::FactorGroupType>>(factorGroupValue);
    auto split = splitOutDeviceLocalFactors(factorGroup);
    if (failed(split)) {
      anchorOp->emitRemark() << "inline-device-local-axes: partitioning_axes["
                             << dim << "] couldn't be resolved to raw factors";
      sawUnsupported = true;
      continue;
    }
    auto &[kept, deviceLocalExtent] = *split;
    if (deviceLocalExtent == 1) {
      continue;
    }
    newPartitioningAxes[dim] =
        axis::viewFactorsAsProduct(kept, builder, anchorOp.getLoc());
    changed = true;
  }

  if (changed) {
    anchorOp.getPartitioningAxesMutable().assign(newPartitioningAxes);
  }
  return sawUnsupported;
}

// Strips DeviceLocal factors from partitioning-axis slot `idx`'s OWN content,
// in place: the slot keeps its index (and every dim_partitioning_axes list
// across every operand/result -- and every op's own un-refreshed per-op
// distributed.argument_shardings/output_shardings copy inside the kernel
// body -- keeps referencing that same index unchanged), only the axis.product
// value stored there shrinks to the remaining (non-DeviceLocal) factors, or
// becomes an empty (extent-1) product if none remain.
//
// This mirrors the established rule elsewhere in this file (see
// mapping_lhs/mapping_rhs in the top comment): a slot several dimensions
// share -- e.g. a dot_general's contracting dimension is the very same
// physical/logical axis on both its lhs and rhs kernel operands, so both
// carry the identical dim_partitioning_axes entry for it -- must never be
// dropped and replaced with a fresh, separately-numbered slot. Doing so
// would desync every OTHER reference to the original index (both the
// kernel's own metadata for the sibling dimension, and every individual op's
// own frozen sharding copy inside the body, which this pass never touches),
// turning what was one shared axis into two independently-tracked ones --
// exactly the kind of mismatch that makes LowerKernelsPass ask Shardy for a
// real reshard where none should ever be needed.
//
// Naturally idempotent: a slot already stripped resolves to deviceLocalExtent
// == 1 on a second visit (from a sibling dimension referencing the same
// index) and this returns success with extent 1 without touching it again.
static FailureOr<int64_t>
stripDeviceLocalFactorsFromSlot(DistributedKernelOp kernelOp, int64_t idx) {
  ValueRange partitioningAxes = kernelOp.getPartitioningAxes();
  if (idx < 0 || idx >= static_cast<int64_t>(partitioningAxes.size())) {
    return failure();
  }
  auto factorGroup =
      dyn_cast<TypedValue<axis::FactorGroupType>>(partitioningAxes[idx]);
  if (!factorGroup) {
    return failure();
  }
  auto split = splitOutDeviceLocalFactors(factorGroup);
  if (failed(split)) {
    return failure();
  }
  auto &[kept, deviceLocalExtent] = *split;
  if (deviceLocalExtent == 1) {
    return 1;
  }
  OpBuilder builder(kernelOp.getContext());
  builder.setInsertionPoint(kernelOp);
  axis::ModuleScopeGuard moduleScope(builder);
  Value newSlot = axis::viewFactorsAsProduct(kept, builder, kernelOp.getLoc());
  kernelOp.getPartitioningAxesMutable()
      .slice(static_cast<unsigned>(idx), 1)
      .assign(newSlot);
  return deviceLocalExtent;
}

// Grows every dimension of `kernelOp`'s own external result types using
// output_shardings' DeviceLocalAxis factors, stripping the now-inlined
// DeviceLocal sub-factors from each referenced slot's own content in place
// (see stripDeviceLocalFactorsFromSlot). output_shardings' dim_partitioning_
// axes index lists never change: DistributedKernelOp::verify()'s own
// invariant (checkLocalGlobalBinding in Ops.cpp) recomputes each dimension's
// declared extent by reading the CURRENT content of whatever slots it
// references, so once a slot's DeviceLocal portion is stripped, every
// dimension referencing it -- this one and any sibling that shares the same
// slot -- sees the smaller extent for free. Only the result's own type needs
// an explicit update here, since growth is this op's own responsibility (see
// the file's top comment).
static bool inlineDeviceLocalAxesInKernelResults(DistributedKernelOp kernelOp) {
  bool sawUnsupported = false;
  ArrayRef<IndexedTensorShardingAttr> outputShardings =
      kernelOp.getOutputShardings().getShardings();

  // First pass (read-only): resolve every slot referenced by ANY result's
  // dim_partitioning_axes to its own DeviceLocal extent, from its current,
  // pre-strip content. A slot is often shared across multiple results of
  // the same kernel (e.g. two outputs of a fused kernel sharing the same
  // batch/head sharding on their leading dimensions) -- computing this
  // lazily while growing each result's shape in turn, as a previous version
  // of this function did, would strip a shared slot while processing the
  // first result that references it, making every later result referencing
  // that same slot see it already emptied (extent 1) and silently skip
  // growth it still needs, since each result's own type is independently
  // sized even when several share one metadata slot.
  llvm::DenseMap<int64_t, int64_t> deviceLocalExtentPerSlot;
  llvm::DenseSet<int64_t> unresolvedSlots;
  ValueRange partitioningAxes = kernelOp.getPartitioningAxes();
  for (auto [resultIdx, result] : llvm::enumerate(kernelOp.getReturns())) {
    if (resultIdx >= outputShardings.size()) {
      break;
    }
    for (DenseI64ArrayAttr dimAxes :
         outputShardings[resultIdx].getDimPartitioningAxes()) {
      for (int64_t idx : dimAxes.asArrayRef()) {
        if (deviceLocalExtentPerSlot.count(idx) || unresolvedSlots.count(idx)) {
          continue;
        }
        if (idx < 0 || idx >= static_cast<int64_t>(partitioningAxes.size())) {
          unresolvedSlots.insert(idx);
          continue;
        }
        auto factorGroup =
            dyn_cast<TypedValue<axis::FactorGroupType>>(partitioningAxes[idx]);
        if (!factorGroup) {
          unresolvedSlots.insert(idx);
          continue;
        }
        auto split = splitOutDeviceLocalFactors(factorGroup);
        if (failed(split)) {
          unresolvedSlots.insert(idx);
          continue;
        }
        deviceLocalExtentPerSlot[idx] = split->second;
      }
    }
  }

  for (auto [resultIdx, result] : llvm::enumerate(kernelOp.getReturns())) {
    if (resultIdx >= outputShardings.size()) {
      break;
    }
    auto rankedType = dyn_cast<RankedTensorType>(result.getType());
    if (!rankedType) {
      continue;
    }
    ArrayRef<DenseI64ArrayAttr> dimAxesList =
        outputShardings[resultIdx].getDimPartitioningAxes();
    if (static_cast<int64_t>(dimAxesList.size()) != rankedType.getRank()) {
      continue;
    }

    SmallVector<int64_t> newShape(rankedType.getShape());
    bool shapeChanged = false;

    for (auto [dim, dimAxes] : llvm::enumerate(dimAxesList)) {
      int64_t deviceLocalExtent = 1;
      bool resolvedAll = true;
      for (int64_t idx : dimAxes.asArrayRef()) {
        if (unresolvedSlots.contains(idx)) {
          resolvedAll = false;
          break;
        }
        deviceLocalExtent *= deviceLocalExtentPerSlot.lookup(idx);
      }
      if (!resolvedAll) {
        kernelOp.emitRemark()
            << "inline-device-local-axes: result " << resultIdx
            << "'s dimension " << dim
            << " references a partitioning-axis slot whose provenance "
               "couldn't be resolved";
        sawUnsupported = true;
        continue;
      }
      if (deviceLocalExtent == 1) {
        continue; // nothing to inline for this dimension.
      }
      newShape[dim] *= deviceLocalExtent;
      shapeChanged = true;
    }

    if (shapeChanged) {
      result.setType(
          RankedTensorType::get(newShape, rankedType.getElementType()));
      propagateGrowthThroughAnchors(result);
    }
  }

  // Second pass: strip every resolved, referenced slot's own content exactly
  // once. Order doesn't matter here -- every result's growth above already
  // used each slot's ORIGINAL extent, cached before any stripping happened.
  for (auto &[idx, extent] : deviceLocalExtentPerSlot) {
    (void)extent;
    (void)stripDeviceLocalFactorsFromSlot(kernelOp, idx);
  }

  return sawUnsupported;
}

// Independently reconciles each kernel operand's declared partitioning
// against its own (fixed, only ever touched by LowerKernelsPass) block-
// argument size: for each dimension whose currently-declared slot(s) still
// include a DeviceLocalAxis-provenance factor, computes the operand's
// correct LOCAL size directly from the block-argument's GLOBAL size divided
// by the dimension's REMAINING (non-DeviceLocal) declared extent, as a sanity
// check before stripping that dimension's referenced slot(s) of their
// DeviceLocal sub-factors in place (see stripDeviceLocalFactorsFromSlot).
// argument_shardings' dim_partitioning_axes index lists never change --
// same reasoning as inlineDeviceLocalAxesInKernelResults above.
//
// Deliberately does NOT mutate `operand`'s (or its real producer's) type
// here: every producer kind (Cast, Kernel, Await via its own collective, or
// an AnchorPartitioningOp forwarding one of those) owns its own authoritative
// growth logic elsewhere in this file. Growing it a second time here --
// especially a DistributedAwait, whose collective is processed in a
// separate, LATER walk than kernels -- would race the producer's own
// decision and can corrupt it outright (observed: an Await grown here to a
// size its own collective, once it later runs, has no idea was already
// "decided"). This function's only job is to keep the kernel's OWN
// partitioning-axis slots in sync (drop the now-inlined DeviceLocal
// sub-factors from whatever they reference); the operand's actual type is
// trusted to converge to the identical target size independently, via its
// own producer's growth logic, by the same "operand-identical decisions are
// always consistent" reasoning that makes every cast/anchor in this dialect
// CSE-safe.
//
// Only handles a dimension whose entire currently-declared slot list is
// resolvable to raw factors, and whose global size evenly divides by the
// remaining extent; either failure is reported and left alone (remark-only,
// matching this pass's usual stance).
static bool reconcileKernelOperandBindings(DistributedKernelOp kernelOp) {
  bool sawUnsupported = false;
  ArrayRef<IndexedTensorShardingAttr> argumentShardings =
      kernelOp.getArgumentShardings().getShardings();
  auto &bodyBlock = kernelOp.getBody().front();

  SmallVector<Value> operands(kernelOp.getArguments().begin(),
                              kernelOp.getArguments().end());

  for (auto [argIdx, operand] : llvm::enumerate(operands)) {
    if (argIdx >= argumentShardings.size() ||
        argIdx >= bodyBlock.getNumArguments()) {
      break;
    }
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto blockArgType =
        dyn_cast<RankedTensorType>(bodyBlock.getArgument(argIdx).getType());
    if (!operandType || !blockArgType ||
        operandType.getRank() != blockArgType.getRank()) {
      continue;
    }
    ArrayRef<DenseI64ArrayAttr> dimAxesList =
        argumentShardings[argIdx].getDimPartitioningAxes();
    if (static_cast<int64_t>(dimAxesList.size()) != operandType.getRank()) {
      continue;
    }

    for (auto [dim, dimAxes] : llvm::enumerate(dimAxesList)) {
      // First pass (read-only): resolve the dimension's combined
      // DeviceLocal/remaining extent across all its slots, without
      // mutating anything, so the divisibility check below sees the
      // pre-strip state consistently regardless of slot-processing order.
      int64_t deviceLocalExtent = 1;
      int64_t remainingExtent = 1;
      bool resolvedAll = true;
      ValueRange currentPartitioningAxes = kernelOp.getPartitioningAxes();
      for (int64_t idx : dimAxes.asArrayRef()) {
        if (idx < 0 ||
            idx >= static_cast<int64_t>(currentPartitioningAxes.size())) {
          resolvedAll = false;
          break;
        }
        auto factorGroup = dyn_cast<TypedValue<axis::FactorGroupType>>(
            currentPartitioningAxes[idx]);
        if (!factorGroup) {
          resolvedAll = false;
          break;
        }
        auto split = splitOutDeviceLocalFactors(factorGroup);
        if (failed(split)) {
          resolvedAll = false;
          break;
        }
        auto &[kept, thisSlotDeviceLocalExtent] = *split;
        deviceLocalExtent *= thisSlotDeviceLocalExtent;
        for (auto factor : kept) {
          auto provenance = axis::getFactorProvenanceAxis(factor);
          if (failed(provenance)) {
            resolvedAll = false;
            break;
          }
          if (!isa<ReplicationAxisType>(provenance->getType())) {
            remainingExtent *=
                static_cast<int64_t>(axis::getFactorExtent(factor));
          }
        }
        if (!resolvedAll) {
          break;
        }
      }
      if (!resolvedAll) {
        kernelOp.emitRemark()
            << "inline-device-local-axes: operand " << argIdx << "'s dimension "
            << dim
            << " references a partitioning-axis slot whose provenance "
               "couldn't be resolved";
        sawUnsupported = true;
        continue;
      }
      if (deviceLocalExtent == 1) {
        continue; // nothing to inline for this dimension.
      }
      int64_t globalDim = blockArgType.getDimSize(dim);
      if (remainingExtent == 0 || globalDim % remainingExtent != 0) {
        kernelOp.emitRemark()
            << "inline-device-local-axes: operand " << argIdx << "'s dimension "
            << dim
            << "'s global size isn't evenly divisible by its remaining "
               "(non-device-local) declared extent";
        sawUnsupported = true;
        continue;
      }
      for (int64_t idx : dimAxes.asArrayRef()) {
        // Failure here would contradict the read-only resolution above,
        // which already validated every slot in this same list.
        (void)stripDeviceLocalFactorsFromSlot(kernelOp, idx);
      }
    }
  }

  return sawUnsupported;
}

// Debug/validation check: flags a kernel operand that needs DeviceLocalAxis
// growth per its own argument_shardings but isn't produced by something this
// pass already grows. What matters is not the producer's op kind but whether
// it OWNS an authoritative partitioning binding for its own result --
// PartitioningAnchorOpInterface, implemented by both Cast ops,
// AnchorPartitioningOp, and DistributedKernelOp -- since that's exactly the
// set of ops this file's other functions grow in place. An
// UnrealizedConversionCastOp (a bare kernel-boundary placeholder with no
// binding of its own) or a DistributedAwait (whose collective is processed
// in a separate, later walk) are the two remaining producer kinds this pass
// can still make consistent, so they're accepted too.
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
    if (isa_and_nonnull<PartitioningAnchorOpInterface,
                        UnrealizedConversionCastOp, DistributedAwait>(
            producer)) {
      continue;
    }
    kernelOp.emitRemark()
        << "inline-device-local-axes: operand " << argIdx
        << " needs DeviceLocalAxis growth per its own argument_shardings, "
           "but its producer owns no PartitioningAnchorOpInterface binding "
           "of its own (nor is it an UnrealizedConversionCastOp or "
           "distributed.Await) for this pass to have already grown";
  }
}

// Strips DeviceLocalAxis factors from a collective's own input_mesh/
// output_mesh/reduction_groups/mapping bookkeeping, and grows its own
// output_type/async_handle result (see the file doc comment for the two
// different reasons factors get dropped from mesh/mapping vs.
// reduction_groups). input_object is never touched; it's expected to
// already be grown by whatever produced it.
static bool
inlineDeviceLocalAxesInCollective(DistributedCollectiveOp collectiveOp,
                                  const AxisToDimensionMap &inputAxisToDim,
                                  const AxisToDimensionMap &outputAxisToDim) {
  OpBuilder builder(collectiveOp);
  Location loc = collectiveOp.getLoc();

  // Everything up to (but not including) the region-count-changed rebuild
  // below is pure axis-algebra and belongs at module scope; only the final
  // collective rebuild is a real op that must stay at collectiveOp's own
  // position, so the guard's scope ends before reaching it.
  FailureOr<std::pair<Value, bool>> newInputMesh, newOutputMesh;
  SmallVector<Value> newReductionGroups;
  SmallVector<Region *> keptBodies;
  bool reductionGroupRemoved = false;
  RankedTensorType newOutputType;
  RankedTensorType currentOutputType;
  Value finalMapping;
  {
  axis::ModuleScopeGuard moduleScope(builder);
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

  newInputMesh = stripGroup(collectiveOp.getInputMesh());
  newOutputMesh = stripGroup(collectiveOp.getOutputMesh());
  if (failed(newInputMesh) || failed(newOutputMesh)) {
    collectiveOp.emitRemark() << "inline-device-local-axes: input_mesh/"
                                 "output_mesh couldn't be resolved to raw "
                                 "factors";
    return true;
  }

  ValueRange oldReductionGroups = collectiveOp.getReductionGroups();
  MutableArrayRef<Region> oldBodies = collectiveOp.getReductionBodies();
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
  currentOutputType = dyn_cast<RankedTensorType>(collectiveOp.getOutputType());
  newOutputType = currentOutputType;
  if (currentOutputType) {
    auto grown = computeGrownOutputType(currentOutputType, oldMappingRhs,
                                        outputAxisToDim);
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
  // Every collective is "bookended" by a DistributedCastGlobalToLocalOp by
  // construction: MaterializeDistributedCollectives.cpp always inserts an
  // intermediary cast pair for a collective that would otherwise chain directly
  // off another collective's own await result, so a *chained* (uncast)
  // input_object should never reach this pass. The numeric dimension-size
  // check in rebuildMappingFactorsForTarget is kept as defense in depth
  // regardless -- it fails (remark, non-fatal) rather than silently
  // miscompiling if that invariant is ever violated.
  auto inputObjectType =
      dyn_cast<RankedTensorType>(collectiveOp.getInputObject().getType());

  SmallVector<Value> newMappingLhs, newMappingRhs;
  bool mappingChanged = false;
  bool inputGrowthExpected = false;
  for (size_t i = 0; i < oldMappingLhs.size(); ++i) {
    auto typedLhs = cast<TypedValue<axis::FactorGroupType>>(oldMappingLhs[i]);
    auto typedRhs = cast<TypedValue<axis::FactorGroupType>>(oldMappingRhs[i]);
    auto lhsResult = rebuildMappingFactorsForTarget(
        typedLhs, inputObjectType, builder, loc, inputAxisToDim);
    auto rhsResult = rebuildMappingFactorsForTarget(
        typedRhs, newOutputType, builder, loc, outputAxisToDim);
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
  // its producer had better already own a PartitioningAnchorOpInterface binding
  // reflecting that growth -- the op kind doesn't matter (a real cast, an
  // AnchorPartitioning, or a kernel boundary all qualify identically). A bare
  // DistributedAwait doesn't own such a binding itself (it just forwards its
  // collective's own result), so it's not accepted here.
  if (inputGrowthExpected) {
    Operation *producer = collectiveOp.getInputObject().getDefiningOp();
    if (!isa_and_nonnull<PartitioningAnchorOpInterface>(producer)) {
      collectiveOp.emitRemark()
          << "inline-device-local-axes: input_object needs DeviceLocalAxis "
             "growth per its mapping, but its producer owns no "
             "PartitioningAnchorOpInterface binding of its own this pass "
             "could have already grown -- "
             "MaterializeDistributedCollectives.cpp is expected to bookend "
             "every collective boundary with one of these";
    }
  }

  // Remark-only: a dropped reduction group assumed some producer's own local
  // contraction already reduced over it (see the loop above). That
  // assumption only makes sense if input_object's producer is one this pass
  // could have grown (any PartitioningAnchorOpInterface op), or a
  // DistributedAwait -- unlike the growth check above, an Await is accepted
  // here too, since the assumption traces back through its own collective's
  // own reduction bookkeeping (checked independently when that collective is
  // itself processed), not through any binding the Await would need to own.
  if (reductionAssumedAlreadyLocal) {
    Operation *producer = collectiveOp.getInputObject().getDefiningOp();
    if (!isa_and_nonnull<PartitioningAnchorOpInterface, DistributedAwait>(
            producer)) {
      collectiveOp.emitRemark()
          << "inline-device-local-axes: dropped a reduction group assuming "
             "it was already reduced locally, but input_object's producer "
             "owns no PartitioningAnchorOpInterface binding of its own (nor "
             "is it a distributed.Await), so that assumption may not hold "
             "here";
    }
  }

  bool meshChanged = newInputMesh->second || newOutputMesh->second;
  if (!meshChanged && !reductionGroupRemoved && !reductionContentChanged &&
      !mappingChanged) {
    return false;
  }

  finalMapping = collectiveOp.getMapping();
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
  } // end of module-scoped axis-algebra construction; builder is restored to
    // collectiveOp's own position below for the real op rebuild.

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
      propagateGrowthThroughAnchors(awaitOp.getValue());
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

    // Resolved before any mutation below: once a kernel/cast/anchor strips
    // its own copy of a DeviceLocalAxis's binding, a collective resolving
    // the same value later in this pass run would otherwise find nothing.
    // input_object and output are resolved independently (see
    // AxisToDimensionMap's comment on why they must never be merged).
    struct CollectiveAxisBindings {
      AxisToDimensionMap input;
      AxisToDimensionMap output;
    };
    llvm::DenseMap<Operation *, CollectiveAxisBindings> collectiveAxisBindings;
    moduleOp.walk([&](DistributedCollectiveOp collectiveOp) {
      // async_handle isn't itself a tensor; a collective's result is only
      // ever unwrapped by its own DistributedAwait, so that's not an
      // arbitrary hop to special-case, just how this value is reached.
      AxisToDimensionMap outputBindings;
      for (OpOperand &use : collectiveOp.getAsyncHandle().getUses()) {
        if (auto awaitOp = dyn_cast<DistributedAwait>(use.getOwner())) {
          outputBindings = resolveResultAxisBindings(awaitOp.getValue());
          break;
        }
      }
      collectiveAxisBindings[collectiveOp.getOperation()] = {
          resolveOperandAxisBindings(collectiveOp.getInputObject()),
          std::move(outputBindings)};
    });

    moduleOp.walk([&](Operation *op) {
      if (auto g2l = dyn_cast<DistributedCastGlobalToLocalOp>(op)) {
        if (inlineDeviceLocalAxesInCast(g2l, /*localSideIsOwnResult=*/true)) {
          sawUnsupported = true;
        }
      } else if (auto l2g = dyn_cast<DistributedCastLocalToGlobalOp>(op)) {
        if (inlineDeviceLocalAxesInCast(l2g, /*localSideIsOwnResult=*/false)) {
          sawUnsupported = true;
        }
      } else if (auto anchor = dyn_cast<AnchorPartitioningOp>(op)) {
        if (inlineDeviceLocalAxesInAnchor(anchor)) {
          sawUnsupported = true;
        }
      } else if (auto kernelOp = dyn_cast<DistributedKernelOp>(op)) {
        if (inlineDeviceLocalAxesInKernelResults(kernelOp)) {
          sawUnsupported = true;
        }
        if (reconcileKernelOperandBindings(kernelOp)) {
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
      const CollectiveAxisBindings &bindings =
          collectiveAxisBindings[collectiveOp.getOperation()];
      if (inlineDeviceLocalAxesInCollective(collectiveOp, bindings.input,
                                            bindings.output)) {
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
