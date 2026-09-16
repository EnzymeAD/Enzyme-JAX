#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Utils.h"

/**
 * Establishes the invariant LowerKernelsPass and InlineDeviceLocalAxesPass
 * depend on: every tensor dimension has all Sharded factors bubbled to the
 * major position of that axis, with Local factors an implicit minor-most
 * remainder -- exactly what Shardy's DimensionShardingAttr requires (it has
 * no syntax for a Local factor before, or interposed between, Sharded ones).
 *
 * Why this invariant can be violated in the first place: our search process
 * always assigns DeviceLocalAxis factors to the minor-most position of a
 * logical axis, but a reshape can merge several logical axes onto one
 * tensor/operator axis. Once merged, that axis's own factors -- some
 * Sharded, some DeviceLocal -- sit in whatever order the merge produced,
 * which can put a DeviceLocal factor in the middle. A device's shard of
 * that axis is then "block-strided" (every Nth block, not one contiguous
 * run), and Shardy's own grammar has no way to declare a Sharded factor
 * anywhere but majormost.
 *
 * THE KEY INSIGHT THIS PASS IS BUILT AROUND: take a global tensor `g`. We
 * can relabel its dimensions/subdimensions into a "transposed" view `G` for
 * free, with no data movement, AS LONG AS we never change the relative
 * order of the Local factors among themselves and each Sharded factor
 * moves together with the tensor dimension it shards. `g` and `G` are THE
 * SAME MATERIALIZATION. No separate tensor is ever computed for `G`: only
 * the *declared* factor order attached to a value differs between the two
 * views.
 *
 * The payoff: choose `G` so its Local factors sit at the minor-most
 * position of each dimension. Taking one device's local shard of `G` then
 * puts that device's Local factors at the minor-most position of the local
 * result too -- exactly the contiguous, Shardy-legal layout we need --
 * while the underlying bytes never move and no extra computation happens
 * anywhere.
 *
 * This makes almost the entire fix pure attribute bookkeeping: no new ops,
 * no type changes, nothing physically rewritten. There are exactly two
 * kinds of place where a real difference is unavoidable, plus one op class
 * this pass declines to handle at all:
 *
 * 1. Ordinary ("Conforming") ops -- elementwise, associative reduction,
 *    transpose, dot_general's batch/free/contracting dims -- and a
 *    DistributedKernelOp's own argument_shardings/output_shardings, simply
 *    never need fixing. Relabeling one of their dimensions is free: an
 *    elementwise op just carries the relabeling through to its result, and
 *    an associative reduction doesn't care what order it reduces its inputs
 *    in. So canonicalizing these is a PURE METADATA reorder -- permute each
 *    dimension's dim_partitioning_axes slot-index list so
 *    Sharded-classified slots precede Local-classified ones -- with no
 *    tensor type change and no new stablehlo op.
 * 2. DistributedCastGlobalToLocalOp/CastLocalToGlobalOp get the same kind
 *    of pure-metadata reorder on their local-side partitioning_axes: they
 *    start declaring canonical (G-relative) order in place of the original
 *    (g-relative) order. Free for the same "same materialization" reason
 *    above, with one extra bit of leverage: a cast is this project's own
 *    dialect op, and nothing in this codebase lowers it to real HLO yet --
 *    so whatever eventually does is free to implement the (possibly
 *    strided) access pattern G's declared order calls for, at zero marginal
 *    cost beyond the extraction the cast always had to perform anyway.
 *    Because the rest of the program -- the caller, a real collective, the
 *    DistributedFunctionOp boundary -- still expects the original (g/g')
 *    view, an explicit reconciling DistributedCollectiveOp ("noop
 *    transpose") needs to bookend each cast to bridge the two. That
 *    bookending collective is not yet implemented.
 * 3. A reshape (or any op whose Shardy sharding rule maps one dimension to
 *    more than one factor) strictly INSIDE a kernel
 *    body is the one real exception: relabeling isn't enough, because a
 *    reshape's own numerical meaning IS its dimension order (it's defined
 *    by flattening the tensor row-major and re-splitting it) -- changing
 *    that order isn't just bookkeeping, it changes which physical bytes the
 *    op actually reads or produces. On top of that, a stablehlo.reshape is
 *    processed directly by Shardy's OWN generic reshape-sharding-rule
 *    machinery inside LowerKernels.cpp's Shardy sub-lowering, which this
 *    project doesn't own and can't instruct to trust a declared order that
 *    doesn't match reality. So this is the one place we still insert real
 *    ops: a "clean" split matching the dimension's current boundary, then a
 *    distributed.ManualComputation (this project's own stand-in for Shardy's
 *    sdy.manual_computation, see its own doc comment in Ops.td) whose region
 *    computes the final local-shape result directly -- no transpose, since
 *    Shardy's per-op reshape-rule machinery never even sees the merge once
 *    it's wrapped this way. Applied strictly at that one op's own
 *    operand/result (see canonicalizeOpNeedingLayout). This needs no
 *    dataflow tracking, because nothing upstream of the fixed-up op is ever
 *    touched by this pass, and its operand already arrives canonically
 *    declared thanks to (1)/(2) above.
 *
 * Not yet supported: an op whose sharding rule marks a factor as
 * needing permutation or replication (e.g. stablehlo.convolution's spatial
 * dims, which need a halo-swap/collective-permute, not a free relabeling).
 * These are remarked and the pass fails rather than silently mishandling
 * them.
 *
 * Boundaries: a DistributedKernelOp's own argument_shardings/
 * output_shardings are purely this compiler's internal bookkeeping and are
 * freely reorderable via the same pure-metadata mechanism as (1) -- no
 * physical rewrite of its block-argument type is needed here. A
 * DistributedFunctionOp's own boundary, by contrast, is a real external
 * contract and is deliberately never touched here -- reconciling it with
 * what an actual call site needs is a separate, later concern.
 */

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_CANONICALIZESHARDEDFACTORORDERPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Marks an op as one of this pass's own boundary-canonicalization rewrite
// ops (the split/manual-computation pair canonicalizeDimOrder builds). These
// are deliberately still in "has a dimension mapped to more than one
// factor" form -- that's exactly the split/merge they exist to perform --
// so the classifier walk below must skip them rather than flag them as an
// unhandled multi-factor reshape.
static constexpr llvm::StringLiteral kInternalRewriteMarker =
    "canonicalize_sharded_factor_order.internal";

// Ops with no tensor-typed operand or result carry no sharded/local factor
// structure to reason about (mesh/axis declarations, terminators without
// operands, etc.) -- always fine to skip regardless of op identity.
static bool touchesAnyTensor(Operation *op) {
  auto isTensor = [](Type t) { return isa<RankedTensorType>(t); };
  return llvm::any_of(op->getOperandTypes(), isTensor) ||
         llvm::any_of(op->getResultTypes(), isTensor);
}

// A kernel's own `distributed.argument_shardings`/`output_shardings` are the
// actual place the sandwiching problem can materialize in real IR: each
// entry in `kernelOp.getPartitioningAxes()` is a FactorGroupType (an
// axis.product of one or more axis.factor values, built by
// ClusterDistributedKernels.cpp), so a tensor dimension's
// `dim_partitioning_axes` index list *is* its physical major-to-minor factor
// order. Once ClusterDistributedKernels.cpp fixes that order, nothing
// upstream of this pass ever revisits it -- so it stays whatever order
// Shardy's own per-op sharding rule originally assigned, without regard to
// which factors later get colored DeviceLocalAxis vs. Sharded.
//
// Appends a brand-new single-purpose partitioning-axis slot (an axis.product
// over exactly `factors`) to `kernelOp`'s own operand list and returns its
// index. Used when canonicalization splits a slot whose own factor list
// mixes Sharded and Local factors (the `composite_kernel` shape in
// lower_kernels.mlir) across the canonical shard/local boundary: the two
// halves can no longer share one slot, since `dim_partitioning_axes` can
// only select whole slots, not partial ones.
//
// Built with its own builder positioned right before `kernelOp` in its
// enclosing block, deliberately ignoring whatever builder the caller is
// otherwise using: `kernelOp`'s partitioning_axes are its own operands, so
// the new axis.product must dominate `kernelOp` itself, not merely
// somewhere inside its body region (where the caller's builder for the
// block-argument rewrite actually points).
static int64_t appendNewPartitioningAxisSlot(
    DistributedKernelOp kernelOp,
    ArrayRef<TypedValue<axis::AxisFactorType>> factors) {
  int64_t newIndex =
      static_cast<int64_t>(kernelOp.getPartitioningAxes().size());
  OpBuilder outerBuilder(kernelOp.getContext());
  outerBuilder.setInsertionPoint(kernelOp);
  Value newSlot =
      axis::viewFactorsAsProduct(factors, outerBuilder, kernelOp.getLoc());
  kernelOp.getPartitioningAxesMutable().append(newSlot);
  return newIndex;
}

// Builds an IndexedTensorShardingAttr where dimension `dim` of the tensor
// described by `fullDimAxesList` is replaced by `dimEntries` (one or more
// entries, e.g. N single-factor slots after a split, or one merged slot list
// after a merge), and every other dimension keeps whatever `fullDimAxesList`
// already says. dim_partitioning_axes is positional (one entry per tensor
// dimension), so no index-shift arithmetic is needed -- just concatenation
// around the replaced entries. Used to build self-consistent
// argument_shardings/output_shardings for each of the three ops
// buildSplitTransposeMergeChain inserts, so LowerKernels.cpp's
// constructShardyAttributes (which only annotates ops carrying these attrs)
// treats them like any other op instead of leaving them unannotated to
// Shardy -- the direct cause of the sdy.sharding_constraint legalization
// failure this fixes.
static IndexedTensorShardingAttr buildRankShiftedSharding(
    MLIRContext *ctx, ArrayRef<DenseI64ArrayAttr> fullDimAxesList, int64_t dim,
    ArrayRef<DenseI64ArrayAttr> dimEntries, DenseI64ArrayAttr unreducedAxes) {
  SmallVector<DenseI64ArrayAttr> newList;
  newList.reserve(fullDimAxesList.size() - 1 + dimEntries.size());
  newList.append(fullDimAxesList.begin(), fullDimAxesList.begin() + dim);
  newList.append(dimEntries.begin(), dimEntries.end());
  newList.append(fullDimAxesList.begin() + dim + 1, fullDimAxesList.end());
  return IndexedTensorShardingAttr::get(ctx, newList, unreducedAxes);
}

// Clean-splits `tensorValue` at dimension `dim` into one sub-dimension per
// entry of `extents` (current order, so the split always matches existing
// structure), then wraps that split value in a distributed.ManualComputation
// declaring the currently-Sharded sub-dimensions as manual axes, whose region
// computes the final (merged-back, canonical-order) local result directly --
// no transpose, ever. Also attaches correct distributed.argument_shardings/
// output_shardings to the split op and the manual-computation op (see
// buildRankShiftedSharding), so LowerKernels.cpp's Shardy translation doesn't
// leave them unannotated.
//
// Why a real rewrite is needed at all, given the top-of-file comment's claim
// that a device's own local tile never changes: at the point this pass runs,
// a Sharded factor's extent is still its global-relative (mesh) size, not the
// 1 a single device ends up with, and Shardy's own reshape-sharding-rule
// machinery derives one specific factor decomposition mechanically from the
// concrete shapes at that size -- it can't be told to trust a declared order
// that doesn't match. distributed.ManualComputation sidesteps this rather
// than working around it: wrapping the split value's Sharded sub-dimensions
// as manual axes means Shardy's own per-op reshape-rule machinery never sees
// (and so never has to be fooled by) the merge at all -- only the
// ManualComputation op itself gets converted (by literal inlining, see
// Shardy's own ManualComputationOpPattern), and its region already computes
// with concrete local sizes throughout, using exactly the same axis-extent
// information (axis::getFactorExtent) this pass already has at this stage.
static Value buildManualComputationChain(
    OpBuilder &builder, Location loc, DistributedKernelOp kernelOp,
    Value tensorValue, int64_t dim, ArrayRef<int64_t> extents,
    ArrayRef<bool> isSharded, ArrayRef<int64_t> singleFactorSlot,
    ArrayRef<DenseI64ArrayAttr> fullDimAxesList,
    DenseI64ArrayAttr unreducedAxes, ArrayRef<int64_t> finalDimAxisIndices) {
  MLIRContext *ctx = builder.getContext();
  int64_t n = static_cast<int64_t>(extents.size());
  auto tensorType = cast<RankedTensorType>(tensorValue.getType());
  int64_t rank = tensorType.getRank();

  SmallVector<int64_t> splitShape;
  splitShape.reserve(rank - 1 + n);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == dim) {
      splitShape.append(extents.begin(), extents.end());
    } else {
      splitShape.push_back(tensorType.getDimSize(i));
    }
  }
  auto splitType =
      RankedTensorType::get(splitShape, tensorType.getElementType());
  auto splitOp =
      builder.create<stablehlo::ReshapeOp>(loc, splitType, tensorValue);
  splitOp->setDiscardableAttr(kInternalRewriteMarker, builder.getUnitAttr());
  Value splitVal = splitOp;

  // fullDimAxesList/unreducedAxes: the tensor's own current per-dimension
  // sharding (untouched dims pass through as-is via buildRankShiftedSharding).
  // singleFactorSlot: one dedicated single-factor partitioning-axis slot per
  // raw factor, in current order, built by the caller. Each new sub-dimension
  // gets exactly one slot in its own dim_partitioning_axes entry, so this is
  // trivially "manual axes majormost" no matter which sub-dimensions end up
  // manual -- no permutation is needed on the operand side at all.
  IndexedTensorShardingAttr operandSharding =
      IndexedTensorShardingAttr::get(ctx, fullDimAxesList, unreducedAxes);
  SmallVector<DenseI64ArrayAttr> splitDimEntries;
  splitDimEntries.reserve(n);
  SmallVector<int64_t> manualAxes;
  for (int64_t k = 0; k < n; ++k) {
    splitDimEntries.push_back(DenseI64ArrayAttr::get(ctx, singleFactorSlot[k]));
    if (isSharded[k]) {
      manualAxes.push_back(singleFactorSlot[k]);
    }
  }
  IndexedTensorShardingAttr splitOutputSharding = buildRankShiftedSharding(
      ctx, fullDimAxesList, dim, splitDimEntries, unreducedAxes);
  splitOp->setAttr(
      "distributed.argument_shardings",
      IndexedTensorShardingPerValueAttr::get(ctx, {operandSharding}));
  splitOp->setAttr(
      "distributed.output_shardings",
      IndexedTensorShardingPerValueAttr::get(ctx, {splitOutputSharding}));

  // finalDimAxisIndices is the caller's already-computed final (canonical,
  // Sharded-major) slot list for `dim` (identical to what it assigns back
  // into axisIndices) -- reused as-is so this can't drift from what the
  // caller ends up declaring. This is the manual computation's *result*
  // sharding: the one place a real permutation of slots (not just a
  // pass-through) shows up, since it's this op's own declared boundary that
  // establishes the canonical order everything downstream sees.
  IndexedTensorShardingAttr manualOutputSharding = buildRankShiftedSharding(
      ctx, fullDimAxesList, dim,
      {DenseI64ArrayAttr::get(ctx, finalDimAxisIndices)}, unreducedAxes);

  // Region block-argument shape: splitShape with each manual (Sharded)
  // sub-dimension divided down to its own local size -- always 1, since
  // LowerKernels.cpp sizes that slot's Shardy mesh axis to exactly match the
  // factor's own extent (see constructShardyAttributes). Non-manual
  // (Local) sub-dimensions are left at full extent unchanged.
  SmallVector<int64_t> localSplitShape(splitShape.begin(), splitShape.end());
  for (int64_t k = 0; k < n; ++k) {
    if (isSharded[k]) {
      localSplitShape[dim + k] = 1;
    }
  }

  // Region result shape: the original tensor's own dimension `dim` divided
  // by the product of its Sharded sub-dimensions' extents (i.e. its local
  // size once Shardy divides those factors down) -- every other dimension
  // unchanged.
  int64_t shardedExtentProduct = 1;
  for (int64_t k = 0; k < n; ++k) {
    if (isSharded[k]) {
      shardedExtentProduct *= extents[k];
    }
  }
  SmallVector<int64_t> localMergedShape(tensorType.getShape());
  localMergedShape[dim] /= shardedExtentProduct;

  auto manualOp = builder.create<DistributedManualComputationOp>(
      loc, TypeRange{tensorType}, ValueRange{splitVal},
      builder.getDenseI64ArrayAttr(manualAxes),
      IndexedTensorShardingPerValueAttr::get(ctx, {splitOutputSharding}),
      IndexedTensorShardingPerValueAttr::get(ctx, {manualOutputSharding}));

  Block *body =
      builder.createBlock(&manualOp.getBody(), {},
                          TypeRange{RankedTensorType::get(
                              localSplitShape, tensorType.getElementType())},
                          {loc});
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
  auto localMergeOp = bodyBuilder.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(localMergedShape, tensorType.getElementType()),
      body->getArgument(0));
  localMergeOp->setDiscardableAttr(kInternalRewriteMarker,
                                   builder.getUnitAttr());
  bodyBuilder.create<DistributedYieldOp>(loc, TypeRange{},
                                         ValueRange{localMergeOp});

  return manualOp.getResult(0);
}

// Given a flattened raw-factor list's Sharded/Local classification, returns
// canonicalPositions such that canonicalPositions[k] is which *current*
// position ends up at output position k under a stable partition: every
// currently-sharded factor first (relative order preserved), then every
// currently-local factor (relative order preserved). Also reports whether
// this is already the identity (nothing to rewrite).
static bool
computeCanonicalPositions(ArrayRef<bool> isSharded,
                          SmallVectorImpl<int64_t> &canonicalPositions) {
  int64_t n = static_cast<int64_t>(isSharded.size());
  canonicalPositions.clear();
  canonicalPositions.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    if (isSharded[i]) {
      canonicalPositions.push_back(i);
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    if (!isSharded[i]) {
      canonicalPositions.push_back(i);
    }
  }
  for (int64_t k = 0; k < n; ++k) {
    if (canonicalPositions[k] != k) {
      return false;
    }
  }
  return true;
}

// Flattens a dimension's slot-index list down to raw-factor granularity: a
// single partitioning-axis slot can itself already be a composite product
// mixing Sharded and Local factors (see `composite_kernel` in
// lower_kernels.mlir), so the canonical shard/local boundary may need to cut
// *through* one slot, not just between slots. Shared by both the pure
// metadata reorder (computeCanonicalSlotReorder) and the real-rewrite path
// (canonicalizeDimOrder, for a multi-factor op's own operand/result).
struct FlattenedRawFactors {
  SmallVector<TypedValue<axis::AxisFactorType>> rawFactors;
  SmallVector<int64_t>
      sourceSlot; // which `axisIndices` position each raw factor came from
  SmallVector<bool> isSharded;
  SmallVector<int64_t> extents;
  SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>> slotFactors;
};

static FailureOr<FlattenedRawFactors>
flattenRawFactors(ValueRange partitioningAxes, ArrayRef<int64_t> axisIndices) {
  FlattenedRawFactors result;
  result.slotFactors.resize(axisIndices.size());

  for (auto [slotPos, idx] : llvm::enumerate(axisIndices)) {
    if (idx < 0 || idx >= static_cast<int64_t>(partitioningAxes.size())) {
      return failure();
    }
    auto factorGroup =
        cast<TypedValue<axis::FactorGroupType>>(partitioningAxes[idx]);
    auto factors = axis::getProductProvenanceFactors(factorGroup);
    if (failed(factors)) {
      return failure();
    }
    result.slotFactors[slotPos].assign(factors->begin(), factors->end());
    for (auto factor : *factors) {
      auto provenance = axis::getFactorProvenanceAxis(factor);
      if (failed(provenance)) {
        return failure();
      }
      result.rawFactors.push_back(factor);
      result.sourceSlot.push_back(static_cast<int64_t>(slotPos));
      result.isSharded.push_back(
          isa<LogicalMeshAxisType>(provenance->getType()));
      result.extents.push_back(
          static_cast<int64_t>(axis::getFactorExtent(factor)));
    }
  }
  return result;
}

// Regroups raw factors (now in canonicalPositions order) back into slots: a
// maximal run of consecutive (in the new order) raw factors that all came
// from the same original slot, in that slot's own original relative order,
// can keep using that slot's existing index unchanged. Any other run (a
// composite slot split across the shard/local boundary) needs a brand-new
// slot, built as a pure axis-algebra value (Pure/CSE-eligible, never a
// tensor-typed op) via appendNewPartitioningAxisSlot.
static SmallVector<int64_t>
regroupIntoSlots(DistributedKernelOp kernelOp, const FlattenedRawFactors &flat,
                 ArrayRef<int64_t> axisIndices,
                 ArrayRef<int64_t> canonicalPositions) {
  int64_t n = static_cast<int64_t>(flat.rawFactors.size());
  SmallVector<int64_t> newAxisIndices;
  for (int64_t k = 0; k < n;) {
    int64_t slot = flat.sourceSlot[canonicalPositions[k]];
    SmallVector<TypedValue<axis::AxisFactorType>> run;
    while (k < n && flat.sourceSlot[canonicalPositions[k]] == slot) {
      run.push_back(flat.rawFactors[canonicalPositions[k]]);
      ++k;
    }
    bool isWholeOriginalSlotInOrder =
        run.size() == flat.slotFactors[slot].size() &&
        std::equal(run.begin(), run.end(), flat.slotFactors[slot].begin());
    if (isWholeOriginalSlotInOrder) {
      newAxisIndices.push_back(axisIndices[slot]);
    } else {
      newAxisIndices.push_back(appendNewPartitioningAxisSlot(kernelOp, run));
    }
  }
  return newAxisIndices;
}

// The pure-metadata half of canonicalization (case (1) above (this file's own
// top-of-file comment)/kernel's own boundary): given one dimension's current
// slot-index list, computes the new canonical list -- Sharded-classified slots
// before Local-classified ones -- with NO tensor value, builder, or physical
// rewrite involved at all: just axis-algebra bookkeeping (new axis.product
// values only where a composite slot must be split, never a stablehlo op).
// Returns std::nullopt if already canonical (nothing to do); fails only if a
// slot's provenance can't be resolved.
static FailureOr<std::optional<SmallVector<int64_t>>>
computeCanonicalSlotReorder(DistributedKernelOp kernelOp,
                            ArrayRef<int64_t> axisIndices) {
  if (axisIndices.empty()) {
    return std::optional<SmallVector<int64_t>>(std::nullopt);
  }
  auto flat = flattenRawFactors(kernelOp.getPartitioningAxes(), axisIndices);
  if (failed(flat)) {
    return failure();
  }
  SmallVector<int64_t> canonicalPositions;
  if (computeCanonicalPositions(flat->isSharded, canonicalPositions)) {
    return std::optional<SmallVector<int64_t>>(std::nullopt);
  }
  return std::optional<SmallVector<int64_t>>(
      regroupIntoSlots(kernelOp, *flat, axisIndices, canonicalPositions));
}

// The real-rewrite half (case (3) above (this file's own top-of-file comment):
// a multi-factor op's own operand/result). Returns the value to use going
// forward for this one dimension (unchanged input if already canonical), and
// rewrites `axisIndices` in place to the new canonical list of slot indices.
// Fails (leaving both untouched) only if a slot's provenance can't be resolved
// at all. `fullDimAxesList`/`unreducedAxes` describe the tensor's own *current*
// per-dimension sharding (all dimensions, not just `dim`) -- needed to build
// self-consistent argument_shardings/output_shardings for the inserted
// split/transpose/merge ops via buildRankShiftedSharding.
static FailureOr<Value>
canonicalizeDimOrder(OpBuilder &builder, Location loc,
                     DistributedKernelOp kernelOp, Value tensorValue,
                     int64_t dim, SmallVectorImpl<int64_t> &axisIndices,
                     ArrayRef<DenseI64ArrayAttr> fullDimAxesList,
                     DenseI64ArrayAttr unreducedAxes) {
  if (axisIndices.empty()) {
    return tensorValue;
  }

  auto flat = flattenRawFactors(kernelOp.getPartitioningAxes(), axisIndices);
  if (failed(flat)) {
    return failure();
  }

  SmallVector<int64_t> canonicalPositions;
  if (computeCanonicalPositions(flat->isSharded, canonicalPositions)) {
    return tensorValue;
  }

  int64_t n = static_cast<int64_t>(flat->rawFactors.size());
  SmallVector<int64_t> singleFactorSlot(n);
  for (int64_t k = 0; k < n; ++k) {
    singleFactorSlot[k] = appendNewPartitioningAxisSlot(
        kernelOp,
        ArrayRef<TypedValue<axis::AxisFactorType>>(flat->rawFactors[k]));
  }

  // Unlike the pure-metadata reorder's regroupIntoSlots (which reuses/merges
  // slots to minimize how many new axis.product values get created), the
  // manual-computation output must keep every raw factor on its OWN
  // single-factor slot: distributed.ManualComputation's manual_axes need to
  // resolve to exactly one raw factor each (see its verifier), which a
  // regrouped composite slot spanning a Sharded/Local boundary would violate.
  SmallVector<int64_t> newAxisIndices(n);
  for (int64_t k = 0; k < n; ++k) {
    newAxisIndices[k] = singleFactorSlot[canonicalPositions[k]];
  }

  Value mergedVal = buildManualComputationChain(
      builder, loc, kernelOp, tensorValue, dim, flat->extents, flat->isSharded,
      singleFactorSlot, fullDimAxesList, unreducedAxes, newAxisIndices);

  axisIndices.assign(newAxisIndices.begin(), newAxisIndices.end());

  return mergedVal;
}

// Canonicalizes every dimension of one boundary tensor value (a kernel block
// argument, or a value about to be yielded) against its own
// IndexedTensorShardingAttr. Returns the value to use going forward (may be
// `value` itself if already canonical on every dimension) and rewrites
// `sharding` in place to describe the new order. `sawUnsupported` is set
// (never cleared) if any dimension couldn't be verified/canonicalized.
static Value canonicalizeBoundaryValue(OpBuilder &builder, Location loc,
                                       DistributedKernelOp kernelOp,
                                       Value value,
                                       IndexedTensorShardingAttr &sharding,
                                       bool &sawUnsupported) {
  auto rankedType = dyn_cast<RankedTensorType>(value.getType());
  if (!rankedType) {
    return value;
  }
  ArrayRef<DenseI64ArrayAttr> dimAxesList = sharding.getDimPartitioningAxes();
  if (static_cast<int64_t>(dimAxesList.size()) != rankedType.getRank()) {
    return value;
  }

  Value current = value;
  SmallVector<DenseI64ArrayAttr> newDimAxes(dimAxesList.begin(),
                                            dimAxesList.end());
  bool changed = false;

  for (int64_t dim = 0; dim < rankedType.getRank(); ++dim) {
    SmallVector<int64_t> axisIndices(dimAxesList[dim].asArrayRef());
    if (axisIndices.empty()) {
      continue;
    }
    // Note: even a single slot can itself be a composite product (mixing
    // Sharded and Local factors, e.g. `composite_kernel` in
    // lower_kernels.mlir), so this can't skip on axisIndices.size() <= 1 --
    // canonicalizeDimOrder does its own (accurate) flattened-factor-count
    // check. Pass the CURRENT (possibly already-updated-by-an-earlier-
    // dimension) newDimAxes as context, not the pristine original
    // dimAxesList -- a value with more than one dimension needing
    // canonicalization must see any earlier dimension's already-rewritten
    // slot list when building the inserted ops' own sharding attrs.
    auto rewritten =
        canonicalizeDimOrder(builder, loc, kernelOp, current, dim, axisIndices,
                             newDimAxes, sharding.getUnreducedAxes());
    if (failed(rewritten)) {
      mlir::emitRemark(loc)
          << "canonicalize-sharded-factor-order: dimension " << dim
          << " of a boundary value in kernel " << kernelOp.getOperationName()
          << " references a partitioning-axis slot whose "
             "provenance couldn't be resolved (index out "
             "of range, or not produced by axis.product)";
      sawUnsupported = true;
      continue;
    }
    if (*rewritten != current) {
      current = *rewritten;
      newDimAxes[dim] =
          DenseI64ArrayAttr::get(kernelOp.getContext(), axisIndices);
      changed = true;
    }
  }

  if (changed) {
    sharding = IndexedTensorShardingAttr::get(kernelOp.getContext(), newDimAxes,
                                              sharding.getUnreducedAxes());
  }
  return current;
}

// case (1) above: the pure-metadata reorder, applied to every dimension of
// every value in `attr` -- no tensor type changes, no new stablehlo ops, only
// dim_partitioning_axes slot-index lists change (and, for a composite slot
// that must be split across the shard/local boundary, a new axis.product
// value -- pure axis-algebra, never a tensor-typed op). Used both for a
// DistributedKernelOp's own argument_shardings/output_shardings and for an
// ordinary (Conforming) op's distributed.argument_shardings/output_shardings.
// Returns the original `attr` unchanged if nothing needed reordering.
static IndexedTensorShardingPerValueAttr
canonicalizeShardingMetadata(DistributedKernelOp kernelOp,
                             IndexedTensorShardingPerValueAttr attr,
                             bool &sawUnsupported) {
  MLIRContext *ctx = kernelOp.getContext();
  ArrayRef<IndexedTensorShardingAttr> shardings = attr.getShardings();
  SmallVector<IndexedTensorShardingAttr> newShardings(shardings.begin(),
                                                      shardings.end());
  bool changed = false;

  for (auto [valueIdx, sharding] : llvm::enumerate(shardings)) {
    ArrayRef<DenseI64ArrayAttr> dimAxesList = sharding.getDimPartitioningAxes();
    SmallVector<DenseI64ArrayAttr> newDimAxes(dimAxesList.begin(),
                                              dimAxesList.end());
    bool valueChanged = false;

    for (auto [dim, dimAxes] : llvm::enumerate(dimAxesList)) {
      auto reordered =
          computeCanonicalSlotReorder(kernelOp, dimAxes.asArrayRef());
      if (failed(reordered)) {
        kernelOp.emitRemark()
            << "canonicalize-sharded-factor-order: value " << valueIdx
            << "'s dimension " << dim
            << " references a partitioning-axis slot whose provenance "
               "couldn't be resolved";
        sawUnsupported = true;
        continue;
      }
      if (!reordered->has_value()) {
        continue;
      }
      newDimAxes[dim] = DenseI64ArrayAttr::get(ctx, **reordered);
      valueChanged = true;
    }

    if (valueChanged) {
      newShardings[valueIdx] = IndexedTensorShardingAttr::get(
          ctx, newDimAxes, sharding.getUnreducedAxes());
      changed = true;
    }
  }

  if (!changed) {
    return attr;
  }
  return IndexedTensorShardingPerValueAttr::get(ctx, newShardings);
}

// case (2) above: canonicalizes a cast op's own `partitioning_axes` --
// unlike a DistributedKernelOp's, this is one FactorGroupType operand
// *directly* per tensor dimension (no shared slot pool, no integer-index
// indirection), so a non-canonical dimension is fixed by building one new
// axis.product with the SAME factors reordered (Sharded-classified first),
// and replacing that one operand. Pure axis-algebra: no tensor type change,
// no stablehlo op. This is what declares canonical order (G/L) on the cast
// in place of whatever order it previously carried (g/l) -- free for the
// "same materialization" reason in this file's top-level comment; bridging
// this declaration back to whatever the cast's other end still expects (g or
// g') is a separate, real op (a bookending DistributedCollectiveOp), not
// built here.
template <typename CastOpTy>
static bool canonicalizeCastPartitioningAxes(CastOpTy castOp) {
  bool sawUnsupported = false;
  ValueRange partitioningAxes = castOp.getPartitioningAxes();
  SmallVector<Value> newPartitioningAxes(partitioningAxes.begin(),
                                         partitioningAxes.end());
  bool changed = false;
  OpBuilder builder(castOp);

  for (auto [dim, factorGroupValue] : llvm::enumerate(partitioningAxes)) {
    auto factorGroup =
        cast<TypedValue<axis::FactorGroupType>>(factorGroupValue);
    auto factors = axis::getProductProvenanceFactors(factorGroup);
    if (failed(factors)) {
      castOp->emitRemark()
          << "canonicalize-sharded-factor-order: partitioning_axes[" << dim
          << "] isn't produced by axis.product, so its factor list can't be "
             "resolved";
      sawUnsupported = true;
      continue;
    }

    SmallVector<bool> isSharded;
    isSharded.reserve(factors->size());
    bool resolvedAll = true;
    for (auto factor : *factors) {
      auto provenance = axis::getFactorProvenanceAxis(factor);
      if (failed(provenance)) {
        resolvedAll = false;
        break;
      }
      isSharded.push_back(isa<LogicalMeshAxisType>(provenance->getType()));
    }
    if (!resolvedAll) {
      castOp->emitRemark()
          << "canonicalize-sharded-factor-order: partitioning_axes[" << dim
          << "] contains a factor whose provenance axis couldn't be resolved";
      sawUnsupported = true;
      continue;
    }

    SmallVector<int64_t> canonicalPositions;
    if (computeCanonicalPositions(isSharded, canonicalPositions)) {
      continue;
    }

    SmallVector<TypedValue<axis::AxisFactorType>> reordered;
    reordered.reserve(factors->size());
    for (int64_t pos : canonicalPositions) {
      reordered.push_back((*factors)[pos]);
    }
    newPartitioningAxes[dim] =
        axis::viewFactorsAsProduct(reordered, builder, castOp.getLoc());
    changed = true;
  }

  if (changed) {
    castOp.getPartitioningAxesMutable().assign(newPartitioningAxes);
  }
  return sawUnsupported;
}

// Resolves the CURRENT, actual declared sharding of `value` from its
// producer -- a DistributedKernelOp's own (possibly just-canonicalized)
// argument_shardings if `value` is one of its block arguments, or an
// upstream op's own (possibly just-canonicalized) output_shardings
// otherwise -- rather than trusting a consumer's own separately-attached
// distributed.argument_shardings copy. That copy is set once by
// ClusterDistributedKernels.cpp and is only ever accurate as long as nothing
// upstream has since been rewritten; this pass's own kernel-boundary/
// Conforming-op pure-metadata reorder (case (1) above) does exactly that
// (walked, in program order, strictly before a later reshape can see it),
// so a reshape fed directly by a canonicalized block argument or an earlier
// Conforming op must re-derive its operand's sharding from the producer, not
// from its own stale attribute, or it will canonicalize against an order the
// operand no longer actually has. Returns failure if it can't be resolved
// (the caller falls back to the consumer's own declared value).
static FailureOr<IndexedTensorShardingAttr>
resolveCurrentSharding(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto kernelOp = dyn_cast_or_null<DistributedKernelOp>(
        blockArg.getOwner()->getParentOp());
    if (!kernelOp) {
      return failure();
    }
    ArrayRef<IndexedTensorShardingAttr> shardings =
        kernelOp.getArgumentShardings().getShardings();
    if (blockArg.getArgNumber() >= shardings.size()) {
      return failure();
    }
    return shardings[blockArg.getArgNumber()];
  }
  Operation *def = value.getDefiningOp();
  if (!def) {
    return failure();
  }
  // DistributedManualComputationOp declares output_shardings as a real ODS
  // argument (stored under the plain attribute name "output_shardings", not
  // the "distributed."-prefixed discardable attribute every other op in a
  // kernel body carries), so it needs its own accessor here rather than the
  // generic string lookup below.
  if (auto manualOp = dyn_cast<DistributedManualComputationOp>(def)) {
    ArrayRef<IndexedTensorShardingAttr> shardings =
        manualOp.getOutputShardings().getShardings();
    auto result = dyn_cast<OpResult>(value);
    if (!result || result.getResultNumber() >= shardings.size()) {
      return failure();
    }
    return shardings[result.getResultNumber()];
  }
  auto outputShardings = def->getAttrOfType<IndexedTensorShardingPerValueAttr>(
      "distributed.output_shardings");
  if (!outputShardings) {
    return failure();
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return failure();
  }
  ArrayRef<IndexedTensorShardingAttr> shardings =
      outputShardings.getShardings();
  if (result.getResultNumber() >= shardings.size()) {
    return failure();
  }
  return shardings[result.getResultNumber()];
}

// Canonicalizes one op that genuinely needs a specific factor order on some
// dimension (a multi-factor op -- a reshape splitting or joining axes),
// using its own already-accurate distributed.argument_shardings/
// output_shardings attribute (set once, consistently, by
// ClusterDistributedKernels.cpp) as the source of truth for its operands'
// and results' current per-dimension factor order. Fixes are applied
// directly at this one op -- immediately before it for an operand, or
// immediately after for a result -- without touching anything else in the
// program, so nothing upstream ever needs to be tracked or revisited: `op`'s
// own attrs already describe reality, precisely because this pass never
// rewrites anything except exactly where an op like this one needs it.
//
// Slot-index resolution (appendNewPartitioningAxisSlot/canonicalizeDimOrder
// above) needs the enclosing kernel's own partitioning_axes list; an op with
// no such enclosing kernel can't be resolved this way and is reported
// unsupported by the caller instead.
static bool canonicalizeOpNeedingLayout(Operation *op,
                                        DistributedKernelOp kernelOp) {
  bool sawUnsupported = false;
  MLIRContext *ctx = op->getContext();

  if (auto argShardingsAttr =
          op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
              "distributed.argument_shardings")) {
    ArrayRef<IndexedTensorShardingAttr> argShardings =
        argShardingsAttr.getShardings();
    SmallVector<IndexedTensorShardingAttr> newArgShardings(argShardings.begin(),
                                                           argShardings.end());
    bool changedArgs = false;
    OpBuilder builder(op);

    for (auto [operandIdx, sharding] : llvm::enumerate(argShardings)) {
      if (operandIdx >= op->getNumOperands()) {
        break;
      }
      Value operand = op->getOperand(operandIdx);
      IndexedTensorShardingAttr thisSharding = sharding;
      if (auto resolved = resolveCurrentSharding(operand);
          succeeded(resolved)) {
        thisSharding = *resolved;
      }
      Value canonicalized =
          canonicalizeBoundaryValue(builder, op->getLoc(), kernelOp, operand,
                                    thisSharding, sawUnsupported);
      if (canonicalized == operand && thisSharding == sharding) {
        continue;
      }
      op->setOperand(operandIdx, canonicalized);
      newArgShardings[operandIdx] = thisSharding;
      changedArgs = true;
    }

    if (changedArgs) {
      op->setAttr("distributed.argument_shardings",
                  IndexedTensorShardingPerValueAttr::get(ctx, newArgShardings));
    }
  }

  if (auto outputShardingsAttr =
          op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
              "distributed.output_shardings")) {
    ArrayRef<IndexedTensorShardingAttr> outputShardings =
        outputShardingsAttr.getShardings();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfter(op);

    for (auto [resultIdx, sharding] : llvm::enumerate(outputShardings)) {
      if (resultIdx >= op->getNumResults()) {
        break;
      }
      Value result = op->getResult(resultIdx);
      // Snapshot uses before inserting anything, same reasoning as the
      // operand side would need if it redirected all uses -- here it
      // matters because a result (unlike an operand slot) can have many
      // uses, all of which need to move to the fixed-up value.
      SmallVector<OpOperand *> existingUses;
      for (OpOperand &use : result.getUses()) {
        existingUses.push_back(&use);
      }
      IndexedTensorShardingAttr thisSharding = sharding;
      Value canonicalized =
          canonicalizeBoundaryValue(builder, op->getLoc(), kernelOp, result,
                                    thisSharding, sawUnsupported);
      if (canonicalized == result) {
        continue;
      }
      for (OpOperand *use : existingUses) {
        use->set(canonicalized);
      }
      // Deliberately NOT updating op's own distributed.output_shardings
      // here (unlike the operand-side case above, which DOES update
      // distributed.argument_shardings when its operand is replaced).
      // `op` itself is a real op whose result type Shardy derives
      // mechanically from its own structural sharding rule plus its
      // operands' declared sharding -- op's own attr must keep describing
      // what op ACTUALLY, structurally produces (the natural, pre-fix
      // order), or Shardy's own reshape-rule-derived local type computation
      // for `op` contradicts the (wrongly reassigned) declared order,
      // producing an inconsistent mesh/type error. Only the newly-inserted
      // chain's own final (distributed.ManualComputation) op -- which really
      // does produce the canonical order -- carries that declaration;
      // downstream consumers are redirected to read from it instead.
      // Confirmed empirically: a merge-type multi-factor op whose own
      // declared output_shardings was previously (incorrectly) overwritten to
      // the post-chain value failed Shardy's own lowering with a mesh/type
      // mismatch.
    }
  }

  return sawUnsupported;
}

// Classification of `op` with respect to the canonical sharded-major/
// local-minor invariant, derived from Shardy's own per-op sharding rule
// (mlir::sdy::OpShardingRuleAttr) rather than ad hoc trait/op-identity
// checks. The rule already distinguishes exactly the cases that matter here:
//  - A pass-through factor (present, 1:1, on every tensor it's mapped to) or
//    a reduction factor (operand-only) is safe: canonical order carries
//    through unchanged, or the factor discharges entirely. This covers
//    elementwise ops, stablehlo.transpose, and dot_general's batch/free/
//    contracting dims uniformly, without hardcoding op identity.
//  - A dimension mapped to more than one factor
//  (rule.hasDimensionsWithMultipleFactors())
//    is exactly the reshape "multi-factor" signature -- a real split/join
//    rewrite is needed, whatever the op is.
//  - A permutation or need-replication factor looks like a clean 1:1
//    mapping by arity alone but isn't safe to treat as plain pass-through:
//    e.g. stablehlo.convolution's spatial dims are registered as
//    FactorType::kPermutation in op_sharding_rule_registry.cc precisely
//    because a sharded spatial dim needs a halo-swap/collective-permute, not
//    a free relabeling. Convolution is exactly the kind of op the
//    elementwise/pass-through argument above does *not* apply to.
enum class OpClassification {
  Conforming,    // no rewrite needed
  MultiFactor,   // split/join signature -- needs the reshape-style rewrite
  SpecialFactor, // permutation/need-replication/blocked-propagation factor
  NoRule,        // no sharding rule could be synthesized at all
};

static OpClassification classifyOp(Operation *op) {
  mlir::sdy::OpShardingRuleAttr rule = getOrSynthesizeOpShardingRule(op).rule;
  if (!rule) {
    return OpClassification::NoRule;
  }
  if (rule.hasDimensionsWithMultipleFactors()) {
    return OpClassification::MultiFactor;
  }
  for (int64_t factor = 0, n = rule.getNumFactors(); factor < n; ++factor) {
    if (!rule.isPassThroughFactor(factor) && !rule.isReductionFactor(factor)) {
      return OpClassification::SpecialFactor;
    }
  }
  return OpClassification::Conforming;
}

struct CanonicalizeShardedFactorOrderPass
    : public impl::CanonicalizeShardedFactorOrderPassBase<
          CanonicalizeShardedFactorOrderPass> {
  using CanonicalizeShardedFactorOrderPassBase::
      CanonicalizeShardedFactorOrderPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool sawUnsupported = false;

    // Pre-order: a DistributedKernelOp's own boundary (and any cast op) must
    // be canonicalized BEFORE the ops inside a kernel body are visited, so
    // resolveCurrentSharding (used by the multi-factor/reshape path) sees
    // an already-canonical producer instead of a stale one. MLIR's default
    // walk order is post-order (children before parents), which is the
    // wrong direction for this dependency.
    moduleOp.walk<WalkOrder::PreOrder>(
        [&](Operation *op) {
          // Structural / container ops and axis-algebra bookkeeping
          // (axis.factor, axis.product, axis.map, ...) carry no
          // tensor-dimension ordering semantics of their own -- nothing to
          // check.
          if (dyn_cast<axis::MaybeTemporaryInterface>(op)) {
            return;
          }
          if (isa<ModuleOp, func::FuncOp, func::ReturnOp, stablehlo::ReturnOp,
                  sdy::ReturnOp>(op)) {
            // Region terminators (stablehlo.return/sdy.return, alongside the
            // func one already listed): carry no independent
            // tensor-dimension-ordering semantics of their own, since whatever
            // op owns their enclosing region (stablehlo.reduce, sdy ops, ...)
            // is what's actually classified/rewritten. DistributedYieldOp is
            // handled specially below (not skipped): it's the one place a
            // DistributedKernelOp's own output_shardings gets set.
            return;
          }
          if (auto yieldOp = dyn_cast<DistributedYieldOp>(op)) {
            // Sets the enclosing DistributedKernelOp's own output_shardings by
            // resolving each yielded operand's ACTUAL current sharding (via
            // resolveCurrentSharding), rather than independently re-deriving a
            // "canonical" declaration for the kernel's own boundary the way
            // argument_shardings can be (see the kernel-arg case below). This
            // must happen AFTER the body's ops are visited (guaranteed by
            // program order within this block, since a yield is always a
            // terminator, visited last among its siblings under pre-order) --
            // otherwise, if the yielded value traces through a multi-factor
            // fix (a merge, which builds brand-new slots independent of
            // anything computed here), a separately/eagerly pure-metadata-
            // reordered output_shardings would reference DIFFERENT (though
            // structurally equivalent) slot indices than what the fixed-up
            // value actually ends up declaring -- a real mismatch
            // constructShardyAttributes would see as two different shardings
            // for the same value. Confirmed empirically before this fix.
            auto kernelOp =
                dyn_cast_or_null<DistributedKernelOp>(op->getParentOp());
            if (!kernelOp) {
              return;
            }
            MLIRContext *ctx = op->getContext();
            ArrayRef<IndexedTensorShardingAttr> existing =
                kernelOp.getOutputShardings().getShardings();
            SmallVector<IndexedTensorShardingAttr> resultShardings;
            resultShardings.reserve(op->getNumOperands());
            for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
              auto resolved = resolveCurrentSharding(operand);
              if (failed(resolved)) {
                // Benign case: a value with no dimensions to canonicalize in
                // the first place (e.g. a rank-0 stablehlo.constant yielded
                // directly -- constants never carry distributed.output_
                // shardings, since they're identical on every device) simply
                // has nothing for resolveCurrentSharding to have found. Keep
                // the existing (already-empty) declaration silently instead
                // of flagging it as unsupported.
                bool benign = idx < existing.size() &&
                              existing[idx].getDimPartitioningAxes().empty();
                if (!benign) {
                  op->emitRemark()
                      << "canonicalize-sharded-factor-order: yielded operand "
                      << idx
                      << "'s current sharding couldn't be resolved from its "
                         "producer (neither a block argument of this kernel "
                         "nor the result of an op carrying "
                         "distributed.output_shardings)";
                  sawUnsupported = true;
                }
                // Fall back to whatever was already declared rather than an
                // empty placeholder -- better to keep a possibly-stale-but-
                // structurally-valid attribute than an empty one.
                resultShardings.push_back(
                    idx < existing.size()
                        ? existing[idx]
                        : IndexedTensorShardingAttr::get(
                              ctx, {}, DenseI64ArrayAttr::get(ctx, {})));
                continue;
              }
              resultShardings.push_back(*resolved);
            }
            kernelOp.setOutputShardingsAttr(
                IndexedTensorShardingPerValueAttr::get(ctx, resultShardings));
            return;
          }
          if (auto kernelOp = dyn_cast<DistributedKernelOp>(op)) {
            // case (1) above: a DistributedKernelOp's own boundary is purely
            // this compiler's own internal organizational unit -- freely
            // reorderable via the pure-metadata mechanism, no physical rewrite
            // of its block-argument type needed (see this file's top-level
            // comment for why that was tried and rejected in an earlier
            // session). DistributedFunctionOp, by contrast, is a real external
            // boundary and is deliberately left out of this walk entirely (its
            // own argument_shardings/output_shardings are never touched). Only
            // argument_shardings is handled here -- output_shardings is set by
            // the DistributedYieldOp case above, once the body has actually
            // been processed.
            bool sawUnsupportedHere = false;
            kernelOp.setArgumentShardingsAttr(canonicalizeShardingMetadata(
                kernelOp, kernelOp.getArgumentShardings(), sawUnsupportedHere));
            if (sawUnsupportedHere) {
              sawUnsupported = true;
            }
            return;
          }
          if (isa<DistributedFunctionOp, DistributedAwait,
                  UnrealizedConversionCastOp, DistributedCollectiveOp,
                  DistributedManualComputationOp>(op)) {
            // DistributedFunctionOp: a real external boundary, deliberately
            // untouched (see above). DistributedAwait just unwraps an
            // already-computed async handle -- a no-op on the payload, per its
            // own doc comment. UnrealizedConversionCastOp is a temporary
            // type-reconciliation marker inserted by earlier passes (e.g.
            // ClusterDistributedKernels.cpp) for a completely separate concern.
            // DistributedCollectiveOp's own mesh operands are flat, mesh-space
            // groupings rather than one-per-tensor-dimension, so it has no
            // positional structure this pass can act on -- whatever correctness
            // is needed around a collective lives entirely in its bookending
            // casts. DistributedManualComputationOp is this pass's own
            // construction (buildManualComputationChain): its argument_
            // shardings/output_shardings are already canonical by
            // construction, and it has no Shardy sharding rule of its own for
            // classifyOp to consult (it isn't a stablehlo/sdy op), so it must
            // be skipped explicitly rather than falling into the "no rule
            // could be synthesized" exotic-op failure below.
            return;
          }
          if (auto g2l = dyn_cast<DistributedCastGlobalToLocalOp>(op)) {
            // case (2) above: declares canonical (G/L) order on the cast's own
            // partitioning_axes -- pure axis-algebra, no tensor type change.
            // The bookending reconciling collective (bridging g<->G) is
            // inserted separately, not here.
            if (canonicalizeCastPartitioningAxes(g2l)) {
              sawUnsupported = true;
            }
            return;
          }
          if (auto l2g = dyn_cast<DistributedCastLocalToGlobalOp>(op)) {
            if (canonicalizeCastPartitioningAxes(l2g)) {
              sawUnsupported = true;
            }
            return;
          }
          if (op->hasAttr(kInternalRewriteMarker)) {
            // This pass's own split/merge reshape -- deliberately still
            // compound-factor shaped, not an unhandled case.
            return;
          }
          if (!touchesAnyTensor(op)) {
            return;
          }

          switch (classifyOp(op)) {
          case OpClassification::Conforming: {
            // case (1) above: pure metadata reorder, same mechanism as a
            // kernel's own boundary above. A Conforming op's own local-type
            // computation, once handed to Shardy, doesn't depend on a
            // dimension's internal factor order: matching axis symbols on
            // multiple operands stay correctly paired after a uniform
            // relabel, and a reduction's Local factors reduce internally
            // regardless of their order.
            auto kernelOp = op->getParentOfType<DistributedKernelOp>();
            if (!kernelOp) {
              return;
            }
            if (auto argShardings =
                    op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
                        "distributed.argument_shardings")) {
              bool sawUnsupportedHere = false;
              auto newAttr = canonicalizeShardingMetadata(
                  kernelOp, argShardings, sawUnsupportedHere);
              if (newAttr != argShardings) {
                op->setAttr("distributed.argument_shardings", newAttr);
              }
              if (sawUnsupportedHere) {
                sawUnsupported = true;
              }
            }
            if (auto outputShardings =
                    op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
                        "distributed.output_shardings")) {
              bool sawUnsupportedHere = false;
              auto newAttr = canonicalizeShardingMetadata(
                  kernelOp, outputShardings, sawUnsupportedHere);
              if (newAttr != outputShardings) {
                op->setAttr("distributed.output_shardings", newAttr);
              }
              if (sawUnsupportedHere) {
                sawUnsupported = true;
              }
            }
            return;
          }
          case OpClassification::MultiFactor: {
            // The split/join signature -- routed by the rule's own structure,
            // not by checking isa<stablehlo::ReshapeOp>, since any op producing
            // a multi-factor rule needs the same treatment. This is the one
            // case that genuinely needs a fix, applied directly at `op` itself.
            auto kernelOp = op->getParentOfType<DistributedKernelOp>();
            if (!kernelOp) {
              op->emitRemark()
                  << op->getName()
                  << ": has a dimension mapped to more than one factor "
                     "(split/join), but isn't inside a distributed kernel, so "
                     "its current per-dimension factor order can't be resolved";
              sawUnsupported = true;
              return;
            }
            if (canonicalizeOpNeedingLayout(op, kernelOp)) {
              sawUnsupported = true;
            }
            return;
          }
          case OpClassification::SpecialFactor:
            // Needs real communication (a halo-swap/collective-permute), not a
            // free relabeling -- out of scope for this pass.
            op->emitRemark()
                << op->getName()
                << ": has a factor requiring permutation or full replication "
                   "(e.g. a windowed/neighborhood-dependent op such as "
                   "convolution) -- not yet supported";
            sawUnsupported = true;
            return;
          case OpClassification::NoRule:
            break; // fall through to the exotic-op check below
          }

          if (isa<sdy::ReshardOp>(op)) {
            op->emitRemark() << op->getName()
                             << ": collective canonicalizing rewrite not yet "
                                "implemented";
            sawUnsupported = true;
            return;
          }

          op->emitRemark() << op->getName() << ": unsupported by "
                           << getArgument()
                           << ", needs an explicit rewrite rule (no Shardy "
                              "sharding rule could be synthesized)";
          sawUnsupported = true;
        });

    if (sawUnsupported) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
