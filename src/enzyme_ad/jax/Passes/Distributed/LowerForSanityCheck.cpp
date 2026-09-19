#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "src/enzyme_ad/jax/Utils.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_LOWERFORSANITYCHECKPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

using TV_AxisFactor = TypedValue<axis::AxisFactorType>;
using TV_FactorGroup = TypedValue<axis::FactorGroupType>;

// One split contributed by a physical mesh axis to a single original tensor
// dimension of a Cast op, in the major-first order axis::
// getProductProvenanceFactors already returns factors in.
struct AxisSplit {
  size_t axisIndex;
  int64_t extent;
};

// Where one dimension of an "expanded" (canonical [meshExtents...,
// localDims...] shaped) tensor came from: either a real physical mesh axis
// (by index into the module's PhysicalMesh) or one of the value's own
// original tensor dimensions, kept in its original relative order.
struct DimLabel {
  bool isMeshAxis;
  size_t index; // mesh axis index, or original-tensor-dimension index.

  int64_t canonicalPosition(size_t numMeshAxes) const {
    return isMeshAxis ? static_cast<int64_t>(index)
                      : static_cast<int64_t>(numMeshAxes + index);
  }
};

Value buildZeroConstant(OpBuilder &builder, Location loc,
                        RankedTensorType type) {
  return builder.create<stablehlo::ConstantOp>(
      loc, DenseElementsAttr::get(type, builder.getZeroAttr(
                                            type.getElementType())));
}

Value buildIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  auto type = RankedTensorType::get({}, builder.getI64Type());
  return builder.create<stablehlo::ConstantOp>(
      loc, DenseElementsAttr::get(type, builder.getI64IntegerAttr(value)));
}


// Places `v`'s own dims (each described by `labels`, one per dim of `v`) at
// their canonical [meshExtents..., <v's own local dims, in original order>]
// positions, broadcasting in full extent for any mesh axis `v` doesn't
// already vary along -- the same "physical axis absent means uniformly
// replicated" reading used throughout this pass (see the pass's own
// top-of-file rationale). broadcast_in_dim requires a strictly increasing
// dimension correspondence, so dims not already in canonical order are
// transposed into place first.
Value placeIntoCanonical(OpBuilder &builder, Location loc, Value v,
                         ArrayRef<DimLabel> labels,
                         ArrayRef<int64_t> meshExtents) {
  size_t numMeshAxes = meshExtents.size();
  SmallVector<int64_t> order(labels.size());
  for (size_t i = 0; i < labels.size(); ++i)
    order[i] = i;
  llvm::stable_sort(order, [&](int64_t a, int64_t b) {
    return labels[a].canonicalPosition(numMeshAxes) <
           labels[b].canonicalPosition(numMeshAxes);
  });

  bool alreadySorted = llvm::is_sorted(order);
  Value sortedValue = v;
  SmallVector<DimLabel> sortedLabels(labels.begin(), labels.end());
  if (!alreadySorted) {
    SmallVector<int64_t> permutation(order.begin(), order.end());
    auto vType = cast<RankedTensorType>(v.getType());
    SmallVector<int64_t> transposedShape;
    for (int64_t srcDim : permutation)
      transposedShape.push_back(vType.getDimSize(srcDim));
    auto transposedType =
        RankedTensorType::get(transposedShape, vType.getElementType());
    sortedValue = builder.create<stablehlo::TransposeOp>(
        loc, transposedType, v, permutation);
    sortedLabels.clear();
    for (int64_t srcDim : permutation)
      sortedLabels.push_back(labels[srcDim]);
  }

  SmallVector<int64_t> broadcastDims;
  size_t numLocalDims = 0;
  for (const DimLabel &label : sortedLabels) {
    broadcastDims.push_back(label.canonicalPosition(numMeshAxes));
    if (!label.isMeshAxis)
      ++numLocalDims;
  }

  auto sortedType = cast<RankedTensorType>(sortedValue.getType());
  SmallVector<int64_t> finalShape(meshExtents.begin(), meshExtents.end());
  finalShape.resize(numMeshAxes + numLocalDims);
  for (auto [dimIdx, label] : llvm::enumerate(sortedLabels)) {
    if (!label.isMeshAxis)
      finalShape[numMeshAxes + label.index] = sortedType.getDimSize(dimIdx);
  }
  // Local dims keep their own relative order (they were never reordered
  // above -- only mesh-axis dims may have moved), so `label.index` for the
  // trailing dims is already exactly its position among local dims.

  auto finalType =
      RankedTensorType::get(finalShape, sortedType.getElementType());
  return builder.create<stablehlo::BroadcastInDimOp>(
      loc, finalType, sortedValue, broadcastDims);
}

// Per original tensor dimension: the ordered list of physical-axis splits
// a Cast's partitioning_axes contributes (`splits`), plus the combined
// extent of any ReplicationAxisType factors on that dimension
// (`replicateDivisor`, tracked for diagnostic purposes -- see below). Per
// DistributedCastGlobalToLocalOp/CastLocalToGlobalOp's own type inference
// (inferTensorViewCastResultType in Ops.cpp), which divides/multiplies by
// every factor's extent regardless of provenance, a Replicate factor with
// extent > 1 on a dimension genuinely shrinks/grows that dimension's
// local/global shape, exactly like a physical split does -- unlike a
// physical split, though, correctly expanding/collapsing it requires an
// actual data operation (duplicating one real copy out to `extent` copies,
// or dropping back down to one representative copy), not just a reshape
// dimension-count adjustment. That data operation is not implemented by
// this pass: expandFromFlat/collapseToFlat instead fail loudly the moment
// replicateDivisor != 1 for any dimension, rather than silently emitting a
// reshape with a mismatched element count. A Replicate factor is still
// fully supported elsewhere in this pass -- at a collective's own
// mesh/mapping level (see lowerCollective's resolveSingleAxisProvenance
// and its Replicate<->real handling) -- this limitation is specifically
// about one appearing inside a Cast's own per-tensor-dimension
// partitioning_axes.
struct CastPerDimInfo {
  SmallVector<SmallVector<AxisSplit>> splits;
  SmallVector<int64_t> replicateDivisor;
};

// Resolves one Cast op's positional partitioning_axes into per-dimension
// split/replicate info (see CastPerDimInfo). Anything other than a
// physical or replication factor (LogicalMeshAxisType, DeviceLocalAxisType,
// or a physical factor covering an axis this pass doesn't recognize) means
// the module reached this pass without every kernel/cast fully lowered
// down to physical/replication axes, which is a pipeline-ordering bug this
// pass treats as a hard failure rather than a partial case (see the pass's
// own top-of-file precondition).
FailureOr<CastPerDimInfo>
resolveCastPerDimSplits(ValueRange partitioningAxes,
                        ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                        Operation *diagnosticAnchor) {
  CastPerDimInfo info;
  for (Value dimAxes : partitioningAxes) {
    auto group = cast<TV_FactorGroup>(dimAxes);
    auto factors = axis::getProductProvenanceFactors(group);
    if (failed(factors)) {
      return diagnosticAnchor->emitError()
             << "expected cast partitioning_axes operand to be produced by "
                "axis.product";
    }
    SmallVector<AxisSplit> splits;
    int64_t replicateDivisor = 1;
    for (TV_AxisFactor factor : *factors) {
      auto provenanceAxis = axis::getFactorProvenanceAxis(factor);
      if (failed(provenanceAxis)) {
        return diagnosticAnchor->emitError()
               << "cast partitioning factor has no resolvable provenance "
                  "axis";
      }
      if (isa<ReplicationAxisType>(provenanceAxis->getType())) {
        replicateDivisor *= axis::getFactorExtent(factor);
        continue;
      }
      auto physicalType =
          dyn_cast<PhysicalCommAxisType>(provenanceAxis->getType());
      if (!physicalType) {
        return diagnosticAnchor->emitError()
               << "expected a fully-lowered cast to bound only physical or "
                  "replication axes, found "
               << provenanceAxis->getType()
               << " -- distributed-lower-for-sanity-check must run after "
                  "every logical/device-local axis has been resolved";
      }
      auto meshIdx = llvm::find(meshAxisTypes, physicalType);
      if (meshIdx == meshAxisTypes.end()) {
        return diagnosticAnchor->emitError()
               << "cast partitioning factor's physical axis does not match "
                  "the module's PhysicalMesh";
      }
      splits.push_back(AxisSplit{
          static_cast<size_t>(meshIdx - meshAxisTypes.begin()),
          axis::getFactorExtent(factor)});
    }
    info.splits.push_back(std::move(splits));
    info.replicateDivisor.push_back(replicateDivisor);
  }
  return info;
}

// Walks through pass-through AnchorPartitioningOp edges (which never change
// scope or shape, see Ops.td's own doc comment on that op) to the first op
// that actually carries scope information.
Operation *skipPassThroughAnchors(Value v) {
  Operation *def = v.getDefiningOp();
  while (auto anchor = dyn_cast_or_null<AnchorPartitioningOp>(def)) {
    def = anchor.getInput().getDefiningOp();
  }
  return def;
}

// Finds the CastLocalToGlobalOp bounding `result`, if any, walking forward
// through `result`'s sole-consumer chain past any pass-through
// AnchorPartitioningOp. Shared by kernel- and collective-result resolution:
// both a DistributedKernelOp's own result and a DistributedCollectiveOp's
// Await result are local-scope values that may or may not be immediately
// globalized by a real cast (see this dialect's own local/global scope rule
// of thumb, Ops.td's comment above DistributedCastGlobalToLocalOp) --
// returns null if `result` is never wrapped by one (a legitimate
// pass-through/un-globalized value).
DistributedCastLocalToGlobalOp findBoundingCastLocalToGlobal(Value result) {
  Value cur = result;
  while (true) {
    if (!cur.hasOneUse())
      break;
    Operation *user = *cur.getUsers().begin();
    if (auto anchor = dyn_cast<AnchorPartitioningOp>(user)) {
      cur = anchor.getOutput();
      continue;
    }
    break;
  }
  Operation *soleUser = cur.hasOneUse() ? *cur.getUsers().begin() : nullptr;
  return dyn_cast_or_null<DistributedCastLocalToGlobalOp>(soleUser);
}

// The whole-module lowering state: an ordered walk over the
// DistributedFunctionOp's body, dispatched by op kind, building both a
// plain IRMapping (for values that stay at their original flat shape) and
// a side table of "expanded" ([meshExtents..., localDims...] shaped)
// values for anything a Cast or Collective ever partitions. See this file's
// header comment (in the generated pass description, Passes.td) for why a
// single canonical full-mesh-rank expansion -- rather than only expanding
// each kernel/collective's own relevant axis subset -- is deliberately used
// here: it makes every expanded value directly comparable/reusable across
// kernel and collective boundaries with no reconciliation step, at the cost
// of looping (and broadcasting) over some axes a given kernel doesn't
// actually need. This pass is explicitly not meant to be efficient.
class SanityCheckLowering {
public:
  SanityCheckLowering(OpBuilder &builder, ArrayRef<PhysicalCommAxisType> meshAxisTypes)
      : builder(builder), meshAxisTypes(meshAxisTypes) {
    for (auto axisType : meshAxisTypes)
      meshExtents.push_back(axisType.getExtent());
  }

  // Named to avoid shadowing mlir::failed(FailureOr<...>) inside this
  // class's own member functions, which unqualified name lookup would
  // otherwise prefer over the free function.
  bool sawFailure() const { return hasFailed; }

  // Lowers every op in `block` in order, dispatching kernels/collectives/
  // casts/anchors specially and cloning everything else verbatim. Leaves
  // the terminator (DistributedYieldOp) unlowered -- the caller builds
  // func.return from its (mapped) operands directly.
  void lowerBlock(Block &block) {
    for (Operation &op : block.without_terminator()) {
      if (hasFailed)
        return;
      if (auto kernel = dyn_cast<DistributedKernelOp>(&op)) {
        lowerKernel(kernel);
      } else if (auto castOp = dyn_cast<DistributedCastGlobalToLocalOp>(&op)) {
        // Consumed lazily by whichever kernel/collective needs it -- see
        // getExpandedValue.
        (void)castOp;
      } else if (auto castOp = dyn_cast<DistributedCastLocalToGlobalOp>(&op)) {
        (void)castOp;
      } else if (auto anchor = dyn_cast<AnchorPartitioningOp>(&op)) {
        // Input/output are always identical (Ops.td's own doc comment);
        // just forward.
        mapper.map(anchor.getOutput(), mapper.lookupOrDefault(anchor.getInput()));
      } else if (auto collective = dyn_cast<DistributedCollectiveOp>(&op)) {
        lowerCollective(collective);
      } else if (isa<DistributedAwait>(&op)) {
        // Handled together with its producing Collective in lowerCollective.
        continue;
      } else {
        builder.clone(op, mapper);
      }
    }
  }

  IRMapping &getMapper() { return mapper; }

private:
  OpBuilder &builder;
  ArrayRef<PhysicalCommAxisType> meshAxisTypes;
  SmallVector<int64_t> meshExtents;
  IRMapping mapper;
  // Original SSA value (a Cast's global-scope input, a kernel's local
  // result, or a collective's Await result) -> its canonical expanded new-
  // IR value. Memoized so the same partitioned value is only ever expanded
  // once no matter how many kernels/collectives consume it.
  DenseMap<Value, Value> expandedOf;
  bool hasFailed = false;

  // Resolved per-operand/per-result state for one kernel's lowering; kept
  // at class scope (rather than local to lowerKernel) so buildLoopLevel/
  // computeKernelBodyAndUpdate can take them as plain, non-template
  // ArrayRef parameters.
  struct OperandInfo {
    bool partitioned;
    Value expanded; // valid iff partitioned
    Value flat;    // valid iff !partitioned
    CastPerDimInfo perDimSplits; // iff partitioned
  };
  struct ResultInfo {
    bool partitioned;
    DistributedCastLocalToGlobalOp castOp; // valid iff partitioned
    CastPerDimInfo perDimSplits; // iff partitioned
  };

  size_t numMeshAxes() const { return meshAxisTypes.size(); }

  Operation *fail(Operation *anchor, const Twine &message) {
    anchor->emitError() << message;
    hasFailed = true;
    return nullptr;
  }

  // Resolves `original`'s canonical expanded form, walking through
  // pass-through anchors and (if not already memoized) a bounding
  // CastGlobalToLocalOp. Returns null (and leaves `perDimSplitsOut` empty)
  // if `original` isn't partitioned at all -- a legitimate pass-through
  // value the caller should use at its own flat mapped shape instead.
  Value getExpandedValue(Value original) {
    if (Value memoized = expandedOf.lookup(original))
      return memoized;

    Operation *def = skipPassThroughAnchors(original);
    if (auto castOp = dyn_cast_or_null<DistributedCastGlobalToLocalOp>(def)) {
      Value alreadyExpandedInput = getExpandedValue(castOp.getInput());
      Value expanded;
      if (alreadyExpandedInput) {
        expanded = alreadyExpandedInput;
      } else {
        auto flatType = cast<RankedTensorType>(castOp.getInput().getType());
        auto perDimSplits = resolveCastPerDimSplits(
            castOp.getPartitioningAxes(), meshAxisTypes, castOp);
        if (failed(perDimSplits))
          return fail(castOp, "unresolvable cast"), nullptr;
        Value flat = mapper.lookupOrDefault(castOp.getInput());
        expanded = expandFromFlat(flat, flatType, *perDimSplits);
      }
      expandedOf[castOp.getInput()] = expanded;
      expandedOf[original] = expanded;
      return expanded;
    }
    if (auto castOp = dyn_cast_or_null<DistributedCastLocalToGlobalOp>(def)) {
      // Reached via a CastGlobalToLocal's own input recursing into
      // getExpandedValue (never as a top-level call on a kernel operand or
      // collective input directly, since those are local-scope and would
      // only ever have a CastGlobalToLocal or Await as their own defining
      // op) -- this is the cast-pair "round trip" chaining pattern
      // MaterializeDistributedCollectives.cpp's own comment describes (a
      // pure relabeling, no data movement), so the correct expansion is
      // simply whatever the round trip's own pre-cast local input already
      // resolves to.
      return getExpandedValue(castOp.getInput());
    }
    // Not behind a Cast: either genuinely un-partitioned (pass-through), or
    // it's a kernel result / collective Await result that must already be
    // memoized by the time anything downstream asks for it (program order
    // guarantees the producer is lowered first).
    return nullptr;
  }

  Value expandFromFlat(Value flat, RankedTensorType flatType,
                       const CastPerDimInfo &perDimInfo) {
    SmallVector<int64_t> intermediateShape;
    SmallVector<DimLabel> labels;
    for (auto [d, splits] : llvm::enumerate(perDimInfo.splits)) {
      // See CastPerDimInfo's doc comment: correctly expanding a
      // Replicate-affected dimension needs an actual drop-to-one-copy data
      // operation this pass doesn't implement, so fail loudly rather than
      // emit a reshape with a mismatched element count.
      if (perDimInfo.replicateDivisor[d] != 1) {
        mlir::emitError(flat.getLoc())
            << "distributed-lower-for-sanity-check does not yet support a "
              "ReplicationAxisType factor inside a Cast's own per-tensor-"
              "dimension partitioning_axes (dim " << d << ")";
        hasFailed = true;
        return nullptr;
      }
      int64_t extentProduct = 1;
      for (const AxisSplit &split : splits) {
        intermediateShape.push_back(split.extent);
        labels.push_back(DimLabel{true, split.axisIndex});
        extentProduct *= split.extent;
      }
      int64_t remainder = flatType.getDimSize(d) / extentProduct;
      intermediateShape.push_back(remainder);
      labels.push_back(DimLabel{false, static_cast<size_t>(d)});
    }
    Value reshaped = flat;
    if (intermediateShape != flatType.getShape()) {
      auto intermediateType =
          RankedTensorType::get(intermediateShape, flatType.getElementType());
      reshaped = builder.create<stablehlo::ReshapeOp>(flat.getLoc(),
                                                       intermediateType, flat);
    }
    return placeIntoCanonical(builder, flat.getLoc(), reshaped, labels,
                              meshExtents);
  }

  // Inverse of expandFromFlat: collapses a canonical expanded tensor back
  // down to one Cast's own flat local/global tensor type.
  Value collapseToFlat(Value expanded, RankedTensorType flatType,
                       const CastPerDimInfo &perDimInfo,
                       Location loc) {
    // See CastPerDimInfo's doc comment / expandFromFlat's identical guard:
    // collapsing back through a Replicate-affected dimension needs an
    // actual duplicate-to-`extent`-copies data operation this pass doesn't
    // implement.
    for (int64_t divisor : perDimInfo.replicateDivisor) {
      if (divisor != 1) {
        mlir::emitError(loc)
            << "distributed-lower-for-sanity-check does not yet support a "
              "ReplicationAxisType factor inside a Cast's own per-tensor-"
              "dimension partitioning_axes";
        hasFailed = true;
        return nullptr;
      }
    }
    size_t n = numMeshAxes();
    llvm::SmallBitVector used(n);
    for (const auto &splits : perDimInfo.splits)
      for (const AxisSplit &split : splits)
        used.set(split.axisIndex);

    auto expandedType = cast<RankedTensorType>(expanded.getType());
    SmallVector<int64_t> sliceStarts(expandedType.getRank(), 0);
    SmallVector<int64_t> sliceLimits(expandedType.getShape());
    SmallVector<int64_t> sliceStrides(expandedType.getRank(), 1);
    for (size_t a = 0; a < n; ++a) {
      if (!used.test(a))
        sliceLimits[a] = 1;
    }
    Value sliced = expanded;
    if (sliceLimits != llvm::to_vector(expandedType.getShape())) {
      auto slicedType = RankedTensorType::get(
          SmallVector<int64_t>(sliceLimits), expandedType.getElementType());
      sliced = builder.create<stablehlo::SliceOp>(
          loc, slicedType, expanded, sliceStarts, sliceLimits, sliceStrides);
    }

    SmallVector<int64_t> permutation;
    for (size_t a = 0; a < n; ++a) {
      if (!used.test(a))
        permutation.push_back(a);
    }
    for (auto [d, splits] : llvm::enumerate(perDimInfo.splits)) {
      for (const AxisSplit &split : splits)
        permutation.push_back(split.axisIndex);
      permutation.push_back(n + d);
    }
    auto slicedType = cast<RankedTensorType>(sliced.getType());
    SmallVector<int64_t> transposedShape;
    for (int64_t srcDim : permutation)
      transposedShape.push_back(slicedType.getDimSize(srcDim));
    Value transposed = sliced;
    if (!llvm::is_sorted(permutation)) {
      auto transposedType =
          RankedTensorType::get(transposedShape, slicedType.getElementType());
      transposed = builder.create<stablehlo::TransposeOp>(
          loc, transposedType, sliced, permutation);
    }
    return builder.create<stablehlo::ReshapeOp>(loc, flatType, transposed);
  }

  // Lowers a trivial DistributedKernelOp into N nested stablehlo.while
  // loops (N = the module's total physical mesh rank), each level's
  // induction variable directly serving as that axis's coordinate. See the
  // class doc comment for why every kernel loops over every physical axis
  // rather than only the ones it actually shards along.
  void lowerKernel(DistributedKernelOp kernel) {
    if (!isTriviallyLocalKernel(kernel)) {
      fail(kernel,
          "distributed-lower-for-sanity-check requires every "
          "DistributedKernel to already be trivially local (fully "
          "lowered) -- this is a pipeline-ordering bug, not something this "
          "pass handles partially");
      return;
    }
    Location loc = kernel.getLoc();

    // Resolve each operand: either a Cast-bound partitioned value (in
    // canonical expanded form) or a plain pass-through value.
    SmallVector<OperandInfo> operandInfos;
    for (Value operand : kernel.getArguments()) {
      Operation *def = skipPassThroughAnchors(operand);
      if (auto castOp = dyn_cast_or_null<DistributedCastGlobalToLocalOp>(def)) {
        auto perDimSplits = resolveCastPerDimSplits(
            castOp.getPartitioningAxes(), meshAxisTypes, castOp);
        if (failed(perDimSplits)) {
          hasFailed = true;
          return;
        }
        Value expanded = getExpandedValue(operand);
        if (hasFailed)
          return;
        operandInfos.push_back(
            OperandInfo{true, expanded, nullptr, std::move(*perDimSplits)});
      } else {
        Value expanded = getExpandedValue(operand);
        if (hasFailed)
          return;
        if (expanded) {
          // Reached local scope through an Await, not a Cast: already
          // canonical, and this kernel's own operand type tells us its
          // rank (no per-dim split list to resolve).
          operandInfos.push_back(OperandInfo{true, expanded, nullptr, {}});
        } else {
          operandInfos.push_back(
              OperandInfo{false, nullptr, mapper.lookupOrDefault(operand), {}});
        }
      }
    }

    // Resolve each result's bounding cast (its sole real consumer, walking
    // through pass-through anchors forward).
    SmallVector<ResultInfo> resultInfos;
    for (Value result : kernel.getResults()) {
      if (auto castOp = findBoundingCastLocalToGlobal(result)) {
        auto perDimSplits = resolveCastPerDimSplits(
            castOp.getPartitioningAxes(), meshAxisTypes, castOp);
        if (failed(perDimSplits)) {
          hasFailed = true;
          return;
        }
        resultInfos.push_back(
            ResultInfo{true, castOp, std::move(*perDimSplits)});
      } else {
        resultInfos.push_back(ResultInfo{false, nullptr, {}});
      }
    }

    size_t n = numMeshAxes();
    // Loop-carried accumulator per result: canonical expanded shape for a
    // partitioned result, its own flat local type otherwise (overwritten
    // wholesale each iteration -- correct because a result not partitioned
    // along a given axis is, by this pipeline's own correctness invariant,
    // identical across every coordinate of that axis).
    SmallVector<Value> initCarried;
    for (auto [idx, info] : llvm::enumerate(resultInfos)) {
      if (info.partitioned) {
        SmallVector<int64_t> canonicalShape(meshExtents);
        auto localType = cast<RankedTensorType>(kernel.getResults()[idx].getType());
        canonicalShape.append(localType.getShape().begin(),
                              localType.getShape().end());
        initCarried.push_back(buildZeroConstant(
            builder, loc,
            RankedTensorType::get(canonicalShape, localType.getElementType())));
      } else {
        initCarried.push_back(buildZeroConstant(
            builder, loc,
            cast<RankedTensorType>(kernel.getResults()[idx].getType())));
      }
    }

    SmallVector<Value> allIVs;
    SmallVector<Value> finalCarried = buildLoopLevel(
        kernel, 0, n, initCarried, allIVs, operandInfos, resultInfos);
    if (hasFailed)
      return;

    for (auto [idx, info] : llvm::enumerate(resultInfos)) {
      if (info.partitioned) {
        // Memoize the canonical expanded form under the kernel's own raw
        // result value too (not just the bounding cast's global output):
        // a collective consuming this result directly (local scope, no
        // cast in between) resolves it via getExpandedValue's memo lookup
        // rather than walking through a cast that doesn't exist on that
        // edge.
        expandedOf[kernel.getResults()[idx]] = finalCarried[idx];
        auto globalType =
            cast<RankedTensorType>(info.castOp.getOutput().getType());
        Value flatGlobal = collapseToFlat(finalCarried[idx], globalType,
                                          info.perDimSplits, loc);
        mapper.map(info.castOp.getOutput(), flatGlobal);
      } else {
        mapper.map(kernel.getResults()[idx], finalCarried[idx]);
      }
    }
  }

  // Recursively builds one nested stablehlo.while per remaining physical
  // mesh axis (levels [axisLevel, n)); at axisLevel == n, directly clones
  // the kernel body once (using every collected induction variable) instead
  // of opening another loop.
  SmallVector<Value> buildLoopLevel(DistributedKernelOp kernel, size_t axisLevel,
                                    size_t n, ArrayRef<Value> carried,
                                    SmallVectorImpl<Value> &allIVs,
                                    ArrayRef<OperandInfo> operandInfos,
                                    ArrayRef<ResultInfo> resultInfos) {
    Location loc = kernel.getLoc();
    if (axisLevel == n) {
      return computeKernelBodyAndUpdate(kernel, carried, allIVs, operandInfos,
                                        resultInfos);
    }

    SmallVector<Value> operandsInit;
    operandsInit.push_back(buildIndexConstant(builder, loc, 0));
    operandsInit.append(carried.begin(), carried.end());
    SmallVector<Type> types;
    for (Value v : operandsInit)
      types.push_back(v.getType());

    auto whileOp = builder.create<stablehlo::WhileOp>(loc, types, operandsInit);

    {
      // The guard must wrap createBlock itself: createBlock moves the
      // builder's insertion point to the new block as a side effect, so
      // constructing the guard afterward would capture (and later
      // "restore" to) that new-block position instead of the real outer
      // insertion point.
      OpBuilder::InsertionGuard guard(builder);
      Block *condBlock = builder.createBlock(&whileOp.getCond());
      for (Type t : types)
        condBlock->addArgument(t, loc);
      builder.setInsertionPointToStart(condBlock);
      Value bound = buildIndexConstant(builder, loc, meshExtents[axisLevel]);
      auto cmp = builder.create<stablehlo::CompareOp>(
          loc, condBlock->getArgument(0), bound,
          stablehlo::ComparisonDirection::LT);
      builder.create<stablehlo::ReturnOp>(loc, ValueRange{cmp});
    }

    SmallVector<Value> finalCarried;
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *bodyBlock = builder.createBlock(&whileOp.getBody());
      for (Type t : types)
        bodyBlock->addArgument(t, loc);
      builder.setInsertionPointToStart(bodyBlock);
      Value iv = bodyBlock->getArgument(0);
      SmallVector<Value> innerCarried(bodyBlock->getArguments().drop_front());

      allIVs.push_back(iv);
      SmallVector<Value> newCarried = buildLoopLevel(
          kernel, axisLevel + 1, n, innerCarried, allIVs, operandInfos,
          resultInfos);
      allIVs.pop_back();
      if (hasFailed)
        return {};

      Value one = buildIndexConstant(builder, loc, 1);
      Value ivNext = builder.create<stablehlo::AddOp>(loc, iv, one);
      SmallVector<Value> yielded;
      yielded.push_back(ivNext);
      yielded.append(newCarried.begin(), newCarried.end());
      builder.create<stablehlo::ReturnOp>(loc, yielded);
    }
    for (size_t i = 0; i < carried.size(); ++i)
      finalCarried.push_back(whileOp.getResults()[1 + i]);
    return finalCarried;
  }

  SmallVector<Value> computeKernelBodyAndUpdate(
      DistributedKernelOp kernel, ArrayRef<Value> carried,
      ArrayRef<Value> allIVs, ArrayRef<OperandInfo> operandInfos,
      ArrayRef<ResultInfo> resultInfos) {
    Location loc = kernel.getLoc();
    Value zeroIdx = buildIndexConstant(builder, loc, 0);

    IRMapping bodyMapping;
    Block &body = kernel.getBody().front();
    for (auto [blockArg, info] : llvm::zip_equal(body.getArguments(), operandInfos)) {
      Value local;
      if (info.partitioned) {
        auto localType = cast<RankedTensorType>(blockArg.getType());
        SmallVector<Value> starts(allIVs.begin(), allIVs.end());
        SmallVector<int64_t> sliceShape(numMeshAxes(), 1);
        for (int64_t dim : localType.getShape()) {
          starts.push_back(zeroIdx);
          sliceShape.push_back(dim);
        }
        auto sliceType =
            RankedTensorType::get(sliceShape, localType.getElementType());
        Value slice = builder.create<stablehlo::DynamicSliceOp>(
            loc, sliceType, info.expanded, starts, sliceShape);
        local = builder.create<stablehlo::ReshapeOp>(loc, localType, slice);
      } else {
        local = info.flat;
      }
      bodyMapping.map(blockArg, local);
    }

    for (Operation &op : body.without_terminator())
      builder.clone(op, bodyMapping);
    auto yieldOp = cast<DistributedYieldOp>(body.getTerminator());

    SmallVector<Value> newCarried;
    for (auto [idx, info] : llvm::enumerate(resultInfos)) {
      Value localResult = bodyMapping.lookupOrDefault(yieldOp.getReturns()[idx]);
      if (info.partitioned) {
        auto localType = cast<RankedTensorType>(localResult.getType());
        SmallVector<int64_t> updateShape(numMeshAxes(), 1);
        updateShape.append(localType.getShape().begin(), localType.getShape().end());
        auto updateType =
            RankedTensorType::get(updateShape, localType.getElementType());
        Value update =
            builder.create<stablehlo::ReshapeOp>(loc, updateType, localResult);
        SmallVector<Value> starts(allIVs.begin(), allIVs.end());
        for (size_t i = 0; i < localType.getRank(); ++i)
          starts.push_back(zeroIdx);
        newCarried.push_back(builder.create<stablehlo::DynamicUpdateSliceOp>(
            loc, carried[idx].getType(), carried[idx], update, starts));
      } else {
        newCarried.push_back(localResult);
      }
    }
    return newCarried;
  }

  // Lowers a DistributedCollective + its (unique) DistributedAwait
  // consumer into a reduce-then-map recipe entirely over the canonical
  // expanded representation -- see the pass description in Passes.td for
  // why this never reuses DistributedToHlo.cpp's hardware-collective
  // patterns.
  void lowerCollective(DistributedCollectiveOp collective) {
    Location loc = collective.getLoc();
    Value inputExpanded = getExpandedValue(collective.getInputObject());
    if (hasFailed)
      return;
    if (!inputExpanded) {
      fail(collective,
          "distributed-lower-for-sanity-check requires a collective's "
          "input_object to already be a partitioned (Cast- or Await-"
          "bound) value");
      return;
    }

    size_t n = numMeshAxes();
    // activeMeshDims[i] = which physical axis the i-th leading dim of the
    // running tensor currently represents (labels shrink as axes are
    // reduced/dropped below; local dims never move and are addressed by
    // their own trailing position, tracked separately as `numLocalDims`).
    SmallVector<int64_t> activeMeshDims;
    for (size_t a = 0; a < n; ++a)
      activeMeshDims.push_back(a);
    Value running = inputExpanded;

    auto inputFactors = axis::getProductProvenanceFactors(collective.getInputMesh());
    if (failed(inputFactors)) {
      fail(collective, "collective input_mesh must be produced by axis.product");
      return;
    }

    for (auto [reductionGroup, region] :
        llvm::zip_equal(collective.getReductionGroups(),
                        collective.getReductionBodies())) {
      auto groupFactors =
          axis::getProductProvenanceFactors(cast<TV_FactorGroup>(reductionGroup));
      if (failed(groupFactors)) {
        fail(collective, "reduction group must be produced by axis.product");
        return;
      }

      // One region may fold more than one axis (when a reduction group
      // spans several mesh axis factors), so its kind/identity are
      // resolved once here rather than per axis.
      auto elemType = cast<RankedTensorType>(running.getType()).getElementType();
      auto kind = stablehlo::classifyReduceBlockKind(region.front());
      Value identity = stablehlo::getIdentityValueForReduceKind(builder, loc, elemType, kind);
      if (!identity) {
        fail(collective,
            "distributed-lower-for-sanity-check requires a collective's "
            "reduction body to be a single recognized associative op "
            "(add/mul/min/max/and/or/xor) with a known identity element over "
            "its element type");
        return;
      }

      for (TV_AxisFactor factor : *groupFactors) {
        auto provenance = axis::getFactorProvenanceAxis(factor);
        auto physicalType =
            failed(provenance) ? nullptr
                              : dyn_cast<PhysicalCommAxisType>(provenance->getType());
        if (!physicalType) {
          fail(collective, "expected a fully-lowered physical reduction axis");
          return;
        }
        auto meshIdx = llvm::find(meshAxisTypes, physicalType);
        size_t axisIdx = meshIdx - meshAxisTypes.begin();
        auto dimPos = llvm::find(activeMeshDims, static_cast<int64_t>(axisIdx));
        if (dimPos == activeMeshDims.end()) {
          fail(collective, "reduction axis already consumed");
          return;
        }
        size_t dim = dimPos - activeMeshDims.begin();
        running = foldReduction(running, dim, region.front(), identity, loc);
        activeMeshDims.erase(dimPos);
      }
    }

    // labels[i] tracks, for each remaining leading dim of `running`, either
    // its current physical-axis identity or the target output axis it's
    // been relabeled to by a mapping pair. -1 = not yet labeled by mapping
    // (shouldn't survive to the final placement step).
    SmallVector<int64_t> dimLabels(activeMeshDims);
    auto mapOp = collective.getMapping().getDefiningOp<axis::AxisMapOp>();
    if (!mapOp) {
      fail(collective, "collective mapping must be produced by axis.map");
      return;
    }
    for (auto [lhsGroup, rhsGroup] : mapOp.getTypedMappingPairs()) {
      auto lhsAxis = resolveSingleAxisProvenance(lhsGroup);
      auto rhsAxis = resolveSingleAxisProvenance(rhsGroup);
      if (!lhsAxis && !rhsAxis)
        continue; // Replicate -> Replicate: nothing present either side.
      if (lhsAxis && rhsAxis) {
        auto dimPos = llvm::find(dimLabels, static_cast<int64_t>(lhsAxis->axisIndex));
        if (dimPos == dimLabels.end()) {
          fail(collective, "mapping lhs references an already-consumed axis");
          return;
        }
        *dimPos = static_cast<int64_t>(rhsAxis->axisIndex);
      } else if (lhsAxis && !rhsAxis) {
        // Physical -> Replicate: this axis's data is already uniform
        // across it (consumed by reduction upstream, or asserted redundant
        // by construction) -- drop it by taking its representative slice.
        auto dimPos = llvm::find(dimLabels, static_cast<int64_t>(lhsAxis->axisIndex));
        if (dimPos == dimLabels.end()) {
          fail(collective, "mapping lhs references an already-consumed axis");
          return;
        }
        size_t dim = dimPos - dimLabels.begin();
        running = dropDim(running, dim, loc);
        dimLabels.erase(dimPos);
      }
      // Replicate -> Physical(b) and Replicate -> Replicate need no action
      // now: the final placeIntoCanonical fill-broadcast below materializes
      // any physical axis not already present among dimLabels.
    }

    SmallVector<DimLabel> finalLabels;
    for (int64_t label : dimLabels)
      finalLabels.push_back(DimLabel{true, static_cast<size_t>(label)});
    auto runningType = cast<RankedTensorType>(running.getType());
    size_t numLocalDims = runningType.getRank() - dimLabels.size();
    for (size_t i = 0; i < numLocalDims; ++i)
      finalLabels.push_back(DimLabel{false, i});

    Value finalExpanded = placeIntoCanonical(builder, loc, running, finalLabels, meshExtents);

    // The collective's own DistributedAwait is its sole real consumer (see
    // this dialect's own convention -- Ops.td's rule of thumb above
    // DistributedCastGlobalToLocalOp, and DropIdentityCollectives.cpp's
    // identical assumption). Memoize the expanded form under the Await's
    // result so a downstream kernel/collective consuming it directly (no
    // cast) resolves it via getExpandedValue's memo lookup; additionally,
    // if the Await's result is itself immediately globalized by a real
    // cast (e.g. feeding the function's own return directly), collapse
    // and map that cast's output too -- mirroring lowerKernel's identical
    // handling of its own results, since nothing else in this pass ever
    // clones/maps a Cast op's output otherwise.
    assert(llvm::hasSingleElement(collective->getUsers()) &&
          "a DistributedCollective's async handle must have exactly one "
          "DistributedAwait consumer (see createCollectiveAndAwait)");
    auto await = cast<DistributedAwait>(*collective->getUsers().begin());
    expandedOf[await.getValue()] = finalExpanded;
    if (auto castOp = findBoundingCastLocalToGlobal(await.getValue())) {
      auto perDimSplits = resolveCastPerDimSplits(
          castOp.getPartitioningAxes(), meshAxisTypes, castOp);
      if (failed(perDimSplits)) {
        hasFailed = true;
        return;
      }
      auto globalType = cast<RankedTensorType>(castOp.getOutput().getType());
      Value flatGlobal =
          collapseToFlat(finalExpanded, globalType, *perDimSplits, loc);
      mapper.map(castOp.getOutput(), flatGlobal);
    }
  }

  // Present only for a factor group resolving to a single real
  // PhysicalCommAxisType factor; a Replicate-provenance or empty group
  // resolves to std::nullopt instead (see below), so a non-null result is
  // always physical -- there is no in-band non-physical case to track.
  struct AxisProvenance {
    size_t axisIndex;
  };

  std::optional<AxisProvenance> resolveSingleAxisProvenance(TV_FactorGroup group) {
    auto factors = axis::getProductProvenanceFactors(group);
    if (failed(factors) || factors->empty())
      return std::nullopt;
    auto provenance = axis::getFactorProvenanceAxis(factors->front());
    if (failed(provenance))
      return std::nullopt;
    if (isa<ReplicationAxisType>(provenance->getType()))
      return std::nullopt;
    auto physicalType = dyn_cast<PhysicalCommAxisType>(provenance->getType());
    if (!physicalType)
      return std::nullopt;
    auto meshIdx = llvm::find(meshAxisTypes, physicalType);
    if (meshIdx == meshAxisTypes.end())
      return std::nullopt;
    return AxisProvenance{static_cast<size_t>(meshIdx - meshAxisTypes.begin())};
  }

  // Reduces dimension `dim` of `tensor` via stablehlo.reduce, using
  // `identity` (the reduction body's own identity element, resolved once
  // per reduction body by the caller -- see
  // stablehlo::classifyReduceBlockKind/getIdentityValueForReduceKind) as
  // its init value.
  Value foldReduction(Value tensor, size_t dim, Block &body, Value identity,
                      Location loc) {
    auto type = cast<RankedTensorType>(tensor.getType());
    SmallVector<int64_t> resultShape;
    for (auto [i, extent] : llvm::enumerate(type.getShape()))
      if (i != dim)
        resultShape.push_back(extent);
    auto resultType = RankedTensorType::get(resultShape, type.getElementType());
    auto reduceOp = builder.create<stablehlo::ReduceOp>(
        loc, TypeRange{resultType}, ValueRange{tensor}, ValueRange{identity},
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(dim)}));

    IRMapping bodyMapping;
    body.getParent()->cloneInto(&reduceOp.getBody(), bodyMapping);
    return reduceOp.getResult(0);
  }

  // Static slice picking index `idx` along `dim`, keeping `dim` at size 1
  // (still present in the result's rank).
  Value sliceAtIndex(Value tensor, size_t dim, int64_t idx, Location loc) {
    auto type = cast<RankedTensorType>(tensor.getType());
    SmallVector<int64_t> starts(type.getRank(), 0);
    SmallVector<int64_t> limits(type.getShape());
    SmallVector<int64_t> strides(type.getRank(), 1);
    starts[dim] = idx;
    limits[dim] = idx + 1;
    SmallVector<int64_t> resultShape(type.getRank());
    for (int64_t i = 0; i < type.getRank(); ++i)
      resultShape[i] = limits[i] - starts[i];
    auto resultType = RankedTensorType::get(resultShape, type.getElementType());
    return builder.create<stablehlo::SliceOp>(loc, resultType, tensor, starts,
                                              limits, strides);
  }

  // Drops a size-1 dimension via reshape, first slicing the dim down to its
  // index-0 representative (used for a Physical -> Replicate mapping pair,
  // where every coordinate along `dim` is already known-equal by this
  // pipeline's own correctness invariant).
  Value dropDim(Value tensor, size_t dim, Location loc) {
    Value sliced = sliceAtIndex(tensor, dim, 0, loc);
    auto type = cast<RankedTensorType>(sliced.getType());
    SmallVector<int64_t> newShape;
    for (auto [i, extent] : llvm::enumerate(type.getShape())) {
      if (i != dim)
        newShape.push_back(extent);
    }
    auto newType = RankedTensorType::get(newShape, type.getElementType());
    return builder.create<stablehlo::ReshapeOp>(loc, newType, sliced);
  }
};

// Mirrors ConvertMainToDistributedFunction.cpp's forward conversion, in
// reverse: moves distributed.function's body wholesale into a fresh
// func.func and swaps its terminator. Kept local to this file (rather than
// promoted to Dialect/Distributed/Utilities.h) until a second real caller
// exists.
func::FuncOp convertDistributedFunctionToFunc(DistributedFunctionOp distFn,
                                              OpBuilder &builder) {
  builder.setInsertionPoint(distFn);
  auto funcOp = builder.create<func::FuncOp>(
      distFn.getLoc(), distFn.getSymName(), distFn.getFunctionType(),
      distFn.getSymVisibilityAttr(), distFn.getArgAttrsAttr(),
      distFn.getResAttrsAttr());
  funcOp.getBody().takeBody(distFn.getBody());
  auto &block = funcOp.getBody().front();
  auto yieldOp = cast<DistributedYieldOp>(block.getTerminator());
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(yieldOp);
  builder.create<func::ReturnOp>(yieldOp.getLoc(), yieldOp.getReturns());
  yieldOp.erase();
  distFn.erase();
  return funcOp;
}

struct LowerForSanityCheckPass
    : public impl::LowerForSanityCheckPassBase<LowerForSanityCheckPass> {
  using LowerForSanityCheckPassBase::LowerForSanityCheckPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    const auto &mainFunctionAnalysis = getAnalysis<FindMainFunctionAnalysis>();
    if (!mainFunctionAnalysis.isValid()) {
      emitError(module.getLoc()) << "failed to find main function";
      signalPassFailure();
      return;
    }
    auto distFn = mainFunctionAnalysis.getMainDistributedFunctionOp();
    if (!distFn) {
      emitError(module.getLoc())
          << "distributed-lower-for-sanity-check requires main to still be "
            "a distributed.function -- nothing to lower";
      signalPassFailure();
      return;
    }

    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    if (failed(physicalMesh)) {
      signalPassFailure();
      return;
    }
    SmallVector<PhysicalCommAxisType> meshAxisTypes;
    for (Attribute axisAttr : physicalMesh->getAxesAttr())
      meshAxisTypes.push_back(
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));

    OpBuilder builder(module.getContext());
    Block &mainBlock = distFn.getBody().front();
    builder.setInsertionPointToStart(&mainBlock);

    SanityCheckLowering lowering(builder, meshAxisTypes);
    lowering.lowerBlock(mainBlock);
    if (lowering.sawFailure()) {
      signalPassFailure();
      return;
    }

    auto yieldOp = cast<DistributedYieldOp>(mainBlock.getTerminator());
    SmallVector<Value> newReturns;
    for (Value v : yieldOp.getReturns())
      newReturns.push_back(lowering.getMapper().lookupOrDefault(v));
    builder.setInsertionPoint(yieldOp);
    auto newYield = builder.create<DistributedYieldOp>(yieldOp.getLoc(), newReturns);
    yieldOp.erase();

    // Erase every original distributed.*/axis.* op now that everything has
    // been re-expressed in terms of newly-cloned plain stablehlo ops --
    // walking in reverse so a value's uses are erased before its def.
    SmallVector<Operation *> toErase;
    for (Operation &op : llvm::make_early_inc_range(mainBlock)) {
      if (&op == newYield)
        continue;
      if (isa<DistributedKernelOp, DistributedCastGlobalToLocalOp,
             DistributedCastLocalToGlobalOp, AnchorPartitioningOp,
             DistributedCollectiveOp, DistributedAwait>(op)) {
        toErase.push_back(&op);
      }
    }
    for (Operation *op : llvm::reverse(toErase))
      op->dropAllUses(), op->erase();

    convertDistributedFunctionToFunc(distFn, builder);

    // axis.* metadata ops (LogicalMeshAxes/ReplicationAxis/GetPhysicalMeshAxes/
    // axis.product/axis.map/...) may now be unused module-scope leftovers;
    // a plain DCE pass over the module cleans those up rather than this
    // pass tracking every one it might have consumed.
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
