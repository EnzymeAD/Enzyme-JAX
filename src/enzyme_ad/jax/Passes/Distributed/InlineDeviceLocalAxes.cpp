#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinTypes.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

/**
 * Materializes distributed.DeviceLocalAxis factors into concrete tensor
 * shapes. Inside a DistributedKernelOp's own body, every tensor is already at
 * its "global" size (up until LowerKernelsPass runs) -- a DeviceLocalAxis
 * factor there is an optional marker LowerKernelsPass
 * already knows to skip (isShardableFactor excludes DeviceLocalAxisType
 * unconditionally), so nothing inside a kernel body ever needs touching.
 *
 * The actual problem lives entirely in the "local" view outside a kernel's
 * own body: a DeviceLocalAxis factor represents parallelism the search
 * process (SearchStrategies.cpp) decided to serialize entirely within one
 * device rather than shard across the mesh -- but the concrete tensor types
 * describing that local view (a DistributedKernelOp's own external
 * operand/result types, and the local side of a
 * DistributedCastGlobalToLocalOp/CastLocalToGlobalOp) are not yet grown to
 * include that factor's real extent. This pass grows them.
 *
 * This pass relies on the same precondition CanonicalizeShardedFactorOrderPass
 * establishes: every DeviceLocalAxis factor is contiguous and minor-most on
 * its tensor dimension (a dimension may have a Sharded factor followed by a
 * DeviceLocalAxis factor, but never the reverse) -- without this, growing a
 * dimension by a DeviceLocalAxis factor's extent wouldn't correspond to a
 * clean, unambiguous multiply.
 *
 * Concretely:
 *  - DistributedCastGlobalToLocalOp: its own result (the local side) is grown
 *    per-dimension by that dimension's DeviceLocalAxis factors' extent, and
 *    partitioning_axes has those factors dropped (kept as a Sharded-only,
 *    possibly-empty product) so the op's own global=local*extent relationship
 *    stays internally consistent with the new, larger local type.
 *  - DistributedCastLocalToGlobalOp: symmetric, but its local side is its own
 *    *operand* (fed by whatever produced it, typically a kernel's own result
 *    or another cast) rather than something this op itself produces -- so
 *    only partitioning_axes is rewritten here; the operand's type is expected
 *    to already be grown by its producer (see the growth-consistency check
 *    below).
 *  - DistributedKernelOp: only its own external *result* types are grown
 *    directly (using output_shardings), the same way a cast's own result is.
 *    Its own external *operand* types are never set directly -- like a cast's
 *    local-side operand, they're expected to already be grown by whatever
 *    produces them (a cast's result, or another kernel's result), so a
 *    kernel's own operand growth happens automatically, without recursion,
 *    once casts and kernel results are handled by this same pass.
 *  - A final consistency check flags any kernel operand whose own
 *    argument_shardings predicts real DeviceLocalAxis growth (extent > 1) but
 *    whose producer isn't a DistributedCastGlobalToLocalOp or another
 *    DistributedKernelOp -- i.e. a value this pass has no way to have already
 *    grown, which the "automatic propagation" argument above doesn't cover.
 *
 * Not yet handled: DistributedCollectiveOp. Its own input_mesh/output_mesh/
 * reduction_groups/mapping (built from mapping_lhs/mapping_rhs) need
 * DeviceLocalAxis factors swapped for the corresponding tensor dimension's own
 * (now-grown) factor, which first requires matching a DeviceLocalAxis value
 * appearing in the collective against the same value appearing in whichever
 * cast bookends it (the collective itself has no per-dimension tensor
 * mapping of its own -- see DistributedCollectiveOp's own doc comment).
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

// Same as splitOutDeviceLocalFactors, but for a kernel-style dimension: a list
// of slot indices into `partitioningAxes`, each itself a factor group. Only
// the combined DeviceLocalAxis extent is needed here (kernel-level
// argument_shardings/partitioning_axes are deliberately left untouched, see
// this file's own doc comment -- there's no per-op verifier tying a kernel's
// own declared type to its shardings' factor extents the way a cast has).
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

// Grows every dimension of `kernelOp`'s own external result types using
// output_shardings' DeviceLocalAxis factors -- the result-side counterpart of
// growing a DistributedCastGlobalToLocalOp's own result.
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
    ArrayRef<DenseI64ArrayAttr> dimAxesList =
        outputShardings[resultIdx].getDimPartitioningAxes();
    if (static_cast<int64_t>(dimAxesList.size()) != rankedType.getRank()) {
      continue;
    }

    SmallVector<int64_t> newShape(rankedType.getShape());
    bool changed = false;
    for (auto [dim, dimAxes] : llvm::enumerate(dimAxesList)) {
      auto growth = computeDeviceLocalGrowthFactor(
          kernelOp.getPartitioningAxes(), dimAxes.asArrayRef());
      if (failed(growth)) {
        kernelOp.emitRemark()
            << "inline-device-local-axes: result " << resultIdx
            << "'s dimension " << dim
            << " references a partitioning-axis slot whose provenance "
               "couldn't be resolved";
        sawUnsupported = true;
        continue;
      }
      if (*growth == 1) {
        continue;
      }
      newShape[dim] *= *growth;
      changed = true;
    }
    if (changed) {
      result.setType(
          RankedTensorType::get(newShape, rankedType.getElementType()));
    }
  }
  return sawUnsupported;
}

// Flags (as a remark only -- deliberately non-fatal, see below) a kernel
// operand whose own argument_shardings predicts real DeviceLocalAxis growth
// but whose producer isn't a DistributedCastGlobalToLocalOp or another
// DistributedKernelOp -- i.e. a value this pass has no way of having already
// grown "for free" via ordinary SSA value-sharing (see this file's own doc
// comment for why every other case doesn't need this: casts and kernel
// results are grown directly above, and anything consuming one of those
// already-grown values automatically sees the grown type without any further
// action). The most common real case today is a value fed through
// DistributedCollectiveOp/DistributedAwait, which this pass doesn't yet
// handle (see this file's own doc comment).
//
// Deliberately never treated as a hard failure: SearchStrategies.cpp runs
// this pass internally to score every candidate it explores
// (buildDistributedSearchLoweringPipeline), and collective-fed kernels are
// completely ordinary in real models (e.g. any all-reduce) -- failing the
// pass here would make the search unable to find any usable candidate at
// all for such models, which is a strictly worse outcome than leaving this
// one, already-flagged gap unaddressed a little longer.
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
    if (isa_and_nonnull<DistributedCastGlobalToLocalOp, DistributedKernelOp>(
            producer)) {
      continue;
    }
    kernelOp.emitRemark()
        << "inline-device-local-axes: operand " << argIdx
        << " needs DeviceLocalAxis growth per its own argument_shardings, "
           "but isn't produced by a distributed.CastGlobalToLocal or "
           "distributed.DistributedKernel this pass could have already grown "
           "(likely a not-yet-handled DistributedCollectiveOp/Await chain)";
  }
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
      }
    });

    // Consistency check runs only after every cast/kernel result has already
    // been grown, so every kernel operand this pass could have grown "for
    // free" has actually done so by now. Remark-only (see its own comment) --
    // does not contribute to sawUnsupported.
    moduleOp.walk([&](DistributedKernelOp kernelOp) {
      checkKernelOperandGrowthConsistency(kernelOp);
    });

    if (sawUnsupported) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::enzyme::distributed
