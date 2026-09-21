#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include <functional>
#include <numeric>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveAtoms.h"
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
  int64_t stride;
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

// What a Cast's partitioning_axes does to each original tensor dimension:
// the ordered physical-axis splits (`splits`) and the combined extent of any
// replication factors (`replicateDivisor`).
//
// A replication factor with extent > 1 shrinks or grows the dimension like a
// physical split does (inferTensorViewCastResultType in Ops.cpp divides by
// every factor's extent), but expanding or collapsing it needs a data
// operation this pass does not implement: duplicating one copy out to
// `extent` copies, or dropping back to one. expandFromFlat/collapseToFlat
// fail on replicateDivisor != 1 rather than emit a reshape with the wrong
// element count. Replication factors in a collective's own mapping are
// supported; this only concerns ones inside a Cast.
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
          axis::getFactorExtent(factor), axis::getFactorStride(factor)});
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
        // The argument side may pass through the boundary relayout
        // (reshape/transpose) before reaching the function argument.
        Value input = castOp.getInput();
        while (Operation *def = input.getDefiningOp()) {
          if (!isa<AnchorPartitioningOp, stablehlo::ReshapeOp,
                   stablehlo::TransposeOp>(def))
            break;
          input = def->getOperand(0);
        }
        if (!isa<BlockArgument>(input))
          warnInteriorCast(castOp);
      } else if (auto castOp = dyn_cast<DistributedCastLocalToGlobalOp>(&op)) {
        // The result side may pass through the boundary relayout
        // (reshape/transpose) before the return.
        Value cur = castOp.getOutput();
        bool returned = false;
        while (cur.hasOneUse()) {
          Operation *user = *cur.user_begin();
          if (isa<DistributedYieldOp>(user)) {
            returned = true;
            break;
          }
          if (!isa<AnchorPartitioningOp, stablehlo::ReshapeOp,
                   stablehlo::TransposeOp>(user))
            break;
          cur = user->getResult(0);
        }
        if (!returned)
          warnInteriorCast(castOp);
      } else if (auto anchor = dyn_cast<AnchorPartitioningOp>(&op)) {
        // Input/output are always identical (Ops.td's own doc comment);
        // just forward.
        mapper.map(anchor.getOutput(), mapper.lookupOrDefault(anchor.getInput()));
        if (Value expanded = expandedOf.lookup(anchor.getInput()))
          expandedOf[anchor.getOutput()] = expanded;
      } else if (auto collective = dyn_cast<DistributedCollectiveOp>(&op)) {
        lowerCollective(collective);
      } else if (isa<DistributedAwait>(&op)) {
        // Handled together with its producing Collective in lowerCollective.
        continue;
      } else {
        for (Value operand : op.getOperands())
          (void)getFlatValue(operand);
        builder.clone(op, mapper);
      }
    }
  }

  // Once every kernel is lowered, casts should only mark the function's own
  // argument and result boundaries. One anywhere else means an earlier pass
  // (call inlining, for one) left a scope marker behind. It is still lowered
  // correctly, but is worth surfacing.
  void warnInteriorCast(Operation *cast) {
    cast->emitWarning()
        << "scope cast inside the function body; after lowering, casts "
           "should only appear at function argument and result boundaries";
  }

  // The flat (single-copy) value standing for `original`. A value produced
  // per device with no bounding cast is flattened on first use by taking
  // device 0's copy. That is the whole value only when it is replicated: a
  // collective's mapping says whether its result is (deviceVaryingResults),
  // and a kernel result without a cast is replicated by the earlier passes'
  // construction, since a partitioning cast would otherwise remain.
  Value getFlatValue(Value original) {
    if (mapper.contains(original))
      return mapper.lookup(original);
    if (Value expanded = expandedOf.lookup(original)) {
      if (deviceVaryingResults.contains(original)) {
        fail(original.getDefiningOp(),
             "a collective result that varies across the mesh is used at "
             "global scope without a CastLocalToGlobal, so it has no single "
             "flat value");
        return nullptr;
      }
      auto flatType = cast<RankedTensorType>(original.getType());
      auto expandedType = cast<RankedTensorType>(expanded.getType());
      SmallVector<int64_t> limits(expandedType.getShape());
      for (size_t a = 0; a < numMeshAxes(); ++a)
        limits[a] = 1;
      SmallVector<int64_t> starts(limits.size(), 0), strides(limits.size(), 1);
      Value sliced = builder.create<stablehlo::SliceOp>(
          original.getLoc(),
          RankedTensorType::get(limits, expandedType.getElementType()),
          expanded, starts, limits, strides);
      Value flat = reshapeTo(sliced, flatType.getShape(), original.getLoc());
      mapper.map(original, flat);
      return flat;
    }
    return mapper.lookupOrDefault(original);
  }

  // Copies a flat value to every mesh coordinate: [meshExtents..., dims...].
  Value broadcastToMesh(Value flat, Location loc) {
    auto type = cast<RankedTensorType>(flat.getType());
    SmallVector<int64_t> shape(meshExtents);
    shape.append(type.getShape().begin(), type.getShape().end());
    SmallVector<int64_t> dims;
    for (int64_t i = 0; i < type.getRank(); ++i)
      dims.push_back(numMeshAxes() + i);
    return builder.create<stablehlo::BroadcastInDimOp>(
        loc, RankedTensorType::get(shape, type.getElementType()), flat, dims);
  }

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
  // Collective results that carry different data on different mesh
  // coordinates, as decided by the collective's own mapping. Such a result
  // has no single flat value (see getFlatValue).
  DenseSet<Value> deviceVaryingResults;
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
    DistributedCastLocalToGlobalOp castOp; // null if no cast bounds the result
    CastPerDimInfo perDimSplits;           // iff castOp
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

  // Where a cast's factors live among the atoms of the mesh axes they slice.
  // Each mesh axis is cut only at its own factors' boundaries (factors of one
  // cast are disjoint, so each is exactly one atom); any atom no factor
  // covers is a gap the cast leaves replicated.
  struct CastAtoms {
    SmallVector<SmallVector<axis::AxisAtom>> perAxis;
    SmallVector<size_t> offset; // first atom-dim of each mesh axis.
    size_t total = 0;

    size_t atomDim(const AxisSplit &split) const {
      auto [first, count] = axis::getFactorAtomRange(
          perAxis[split.axisIndex], split.extent, split.stride);
      assert(count == 1 && "cast factors are disjoint");
      return offset[split.axisIndex] + first;
    }
    SmallVector<int64_t> shape() const {
      SmallVector<int64_t> result;
      for (const auto &atoms : perAxis)
        for (const axis::AxisAtom &atom : atoms)
          result.push_back(atom.extent);
      return result;
    }
  };

  FailureOr<CastAtoms> computeCastAtoms(const CastPerDimInfo &info) {
    SmallVector<axis::AtomFactor> factors;
    for (const auto &splits : info.splits)
      for (const AxisSplit &split : splits)
        factors.push_back({split.axisIndex, static_cast<uint64_t>(split.extent),
                           static_cast<uint64_t>(split.stride)});
    SmallVector<uint64_t> axisExtents(meshExtents.begin(), meshExtents.end());
    auto common = axis::computeCommonAtoms(axisExtents, factors);
    if (failed(common))
      return failure();
    CastAtoms result;
    for (size_t a = 0; a < numMeshAxes(); ++a) {
      result.offset.push_back(result.total);
      result.total += common->atomsOf(a).size();
      auto atoms = common->atomsOf(a);
      result.perAxis.emplace_back(atoms.begin(), atoms.end());
    }
    return result;
  }

  // Expands a flat (global-shaped) tensor to the canonical [meshExtents...,
  // localDims...] form: reshape to one dim per factor (with each dimension's
  // unpartitioned remainder last), then place each factor at its mesh atom
  // and broadcast in the atoms no factor covers.
  Value expandFromFlat(Value flat, RankedTensorType flatType,
                       const CastPerDimInfo &perDimInfo) {
    // See CastPerDimInfo's doc comment: correctly expanding a
    // Replicate-affected dimension needs an actual drop-to-one-copy data
    // operation this pass doesn't implement, so fail loudly rather than
    // emit a reshape with a mismatched element count.
    for (auto [d, divisor] : llvm::enumerate(perDimInfo.replicateDivisor)) {
      if (divisor != 1) {
        mlir::emitError(flat.getLoc())
            << "distributed-lower-for-sanity-check does not yet support a "
              "ReplicationAxisType factor inside a Cast's own per-tensor-"
              "dimension partitioning_axes (dim " << d << ")";
        hasFailed = true;
        return nullptr;
      }
    }
    auto castAtoms = computeCastAtoms(perDimInfo);
    if (failed(castAtoms)) {
      mlir::emitError(flat.getLoc())
          << "a cast's factors over one mesh axis are not nested";
      hasFailed = true;
      return nullptr;
    }
    SmallVector<int64_t> intermediateShape, targetPos;
    SmallVector<int64_t> targetShape = castAtoms->shape();
    for (auto [d, splits] : llvm::enumerate(perDimInfo.splits)) {
      int64_t extentProduct = 1;
      for (const AxisSplit &split : splits) {
        intermediateShape.push_back(split.extent);
        targetPos.push_back(castAtoms->atomDim(split));
        extentProduct *= split.extent;
      }
      int64_t remainder = flatType.getDimSize(d) / extentProduct;
      intermediateShape.push_back(remainder);
      targetPos.push_back(castAtoms->total + d);
      targetShape.push_back(remainder);
    }
    Location loc = flat.getLoc();
    Value reshaped = reshapeTo(flat, intermediateShape, loc);
    Value placed = placeAtoms(reshaped, targetPos, targetShape, loc);
    SmallVector<int64_t> canonical(meshExtents);
    canonical.append(targetShape.begin() + castAtoms->total, targetShape.end());
    return reshapeTo(placed, canonical, loc);
  }

  // Inverse of expandFromFlat: collapses a canonical expanded tensor back
  // down to one Cast's own flat local/global tensor type, keeping only index
  // 0 of every atom no factor covers.
  Value collapseToFlat(Value expanded, RankedTensorType flatType,
                       const CastPerDimInfo &perDimInfo, Location loc) {
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
    auto castAtoms = computeCastAtoms(perDimInfo);
    if (failed(castAtoms)) {
      mlir::emitError(loc)
          << "a cast's factors over one mesh axis are not nested";
      hasFailed = true;
      return nullptr;
    }
    auto expandedType = cast<RankedTensorType>(expanded.getType());
    SmallVector<int64_t> atomShape = castAtoms->shape();
    atomShape.append(expandedType.getShape().begin() + numMeshAxes(),
                     expandedType.getShape().end());
    Value atomized = reshapeTo(expanded, atomShape, loc);

    llvm::SmallBitVector used(castAtoms->total);
    for (const auto &splits : perDimInfo.splits)
      for (const AxisSplit &split : splits)
        used.set(castAtoms->atomDim(split));

    SmallVector<int64_t> sliceStarts(atomShape.size(), 0);
    SmallVector<int64_t> sliceLimits(atomShape);
    SmallVector<int64_t> sliceStrides(atomShape.size(), 1);
    for (size_t i = 0; i < castAtoms->total; ++i)
      if (!used.test(i))
        sliceLimits[i] = 1;
    Value sliced = atomized;
    if (sliceLimits != atomShape) {
      sliced = builder.create<stablehlo::SliceOp>(
          loc,
          RankedTensorType::get(sliceLimits, expandedType.getElementType()),
          atomized, sliceStarts, sliceLimits, sliceStrides);
    }

    SmallVector<int64_t> permutation;
    for (size_t i = 0; i < castAtoms->total; ++i)
      if (!used.test(i))
        permutation.push_back(i);
    for (auto [d, splits] : llvm::enumerate(perDimInfo.splits)) {
      for (const AxisSplit &split : splits)
        permutation.push_back(castAtoms->atomDim(split));
      permutation.push_back(castAtoms->total + d);
    }
    auto slicedType = cast<RankedTensorType>(sliced.getType());
    Value transposed = sliced;
    if (!llvm::is_sorted(permutation)) {
      SmallVector<int64_t> transposedShape;
      for (int64_t srcDim : permutation)
        transposedShape.push_back(slicedType.getDimSize(srcDim));
      transposed = builder.create<stablehlo::TransposeOp>(
          loc,
          RankedTensorType::get(transposedShape, slicedType.getElementType()),
          sliced, permutation);
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
              OperandInfo{false, nullptr, getFlatValue(operand), {}});
        }
      }
    }

    // Every result is kept per device (canonical expanded form). A bounding
    // cast, if any, tells how to fold the result back to a global value;
    // without one the result is consumed at local scope, and a flat value is
    // only built if something outside the distributed ops needs it (see
    // getFlatValue).
    SmallVector<ResultInfo> resultInfos;
    for (Value result : kernel.getResults()) {
      auto castOp = findBoundingCastLocalToGlobal(result);
      if (!castOp) {
        resultInfos.push_back(ResultInfo{nullptr, {}});
        continue;
      }
      auto perDimSplits = resolveCastPerDimSplits(castOp.getPartitioningAxes(),
                                                  meshAxisTypes, castOp);
      if (failed(perDimSplits)) {
        hasFailed = true;
        return;
      }
      resultInfos.push_back(ResultInfo{castOp, std::move(*perDimSplits)});
    }

    size_t n = numMeshAxes();
    // Loop-carried accumulator per result, in canonical expanded shape.
    SmallVector<Value> initCarried;
    for (Value result : kernel.getResults()) {
      SmallVector<int64_t> canonicalShape(meshExtents);
      auto localType = cast<RankedTensorType>(result.getType());
      canonicalShape.append(localType.getShape().begin(),
                            localType.getShape().end());
      initCarried.push_back(buildZeroConstant(
          builder, loc,
          RankedTensorType::get(canonicalShape, localType.getElementType())));
    }

    SmallVector<Value> allIVs;
    SmallVector<Value> finalCarried = buildLoopLevel(
        kernel, 0, n, initCarried, allIVs, operandInfos, resultInfos);
    if (hasFailed)
      return;

    for (auto [idx, info] : llvm::enumerate(resultInfos)) {
      // Later kernels and collectives read the kernel's own (local-scope)
      // result through this memo; the bounding cast's global output is only
      // built when a cast exists.
      expandedOf[kernel.getResults()[idx]] = finalCarried[idx];
      if (info.castOp) {
        auto globalType =
            cast<RankedTensorType>(info.castOp.getOutput().getType());
        Value flatGlobal = collapseToFlat(finalCarried[idx], globalType,
                                          info.perDimSplits, loc);
        mapper.map(info.castOp.getOutput(), flatGlobal);
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

  // The device this iteration stands for, numbered the way a real
  // stablehlo.partition_id is: the sum over mesh axes of the axis coordinate
  // times the axis's device-id stride. Kernel bodies use partition_id (e.g.
  // for slices Shardy emits) expecting a different value on every device, but
  // all devices run in this one process, so it is rebuilt from the loop
  // induction variables.
  Value buildDeviceId(Location loc, ArrayRef<Value> allIVs, Type resultType) {
    Value id = buildIndexConstant(builder, loc, 0);
    for (auto [iv, axisType] : llvm::zip_equal(allIVs, meshAxisTypes)) {
      Value stride = buildIndexConstant(builder, loc, axisType.getIdStride());
      Value scaled = builder.create<stablehlo::MulOp>(loc, iv, stride);
      id = builder.create<stablehlo::AddOp>(loc, id, scaled);
    }
    return builder.create<stablehlo::ConvertOp>(loc, resultType, id);
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

    for (Operation &op : body.without_terminator()) {
      if (auto partitionId = dyn_cast<stablehlo::PartitionIdOp>(op)) {
        bodyMapping.map(partitionId.getResult(),
                        buildDeviceId(loc, allIVs, partitionId.getType()));
        continue;
      }
      builder.clone(op, bodyMapping);
    }
    auto yieldOp = cast<DistributedYieldOp>(body.getTerminator());

    SmallVector<Value> newCarried;
    for (auto [idx, info] : llvm::enumerate(resultInfos)) {
      Value localResult = bodyMapping.lookupOrDefault(yieldOp.getReturns()[idx]);
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
    }
    return newCarried;
  }

  // Lowers a DistributedCollective + its (unique) DistributedAwait
  // consumer entirely over the canonical expanded representation (see the
  // pass description in Passes.td for why this never reuses
  // DistributedToHlo.cpp's hardware-collective patterns).
  //
  // The recipe, over the common refinement of the collective's factors
  // (CollectiveAtoms), so a factor may be any sub-range of an axis:
  //  1. Reshape [mesh..., tile...] to one dim per atom.
  //  2. Reduce each reduction group's atom dims.
  //  3. Relabel per mapping pair: lhs atom -> rhs atom, lhs -> replicate drops
  //     the dim to its index-0 representative, replicate -> rhs atom is left
  //     for the broadcast.
  //  4. Transpose into the output's atom order, broadcast in any missing atom
  //     (including whole mesh axes the collective never mentions), and
  //     reshape to [mesh..., output tile...].
  void lowerCollective(DistributedCollectiveOp collective) {
    Location loc = collective.getLoc();
    Value inputExpanded = getExpandedValue(collective.getInputObject());
    if (hasFailed)
      return;
    if (!inputExpanded) {
      // Not produced per device: a replicated value with no cast around it.
      inputExpanded = broadcastToMesh(getFlatValue(collective.getInputObject()),
                                      loc);
      expandedOf[collective.getInputObject()] = inputExpanded;
    }

    // The collective's own DistributedAwait is its sole real consumer (see
    // this dialect's own convention -- Ops.td's rule of thumb above
    // DistributedCastGlobalToLocalOp, and DropIdentityCollectives.cpp's
    // identical assumption).
    assert(llvm::hasSingleElement(collective->getUsers()) &&
           "a DistributedCollective's async handle must have exactly one "
           "DistributedAwait consumer (see createCollectiveAndAwait)");
    auto await = cast<DistributedAwait>(*collective->getUsers().begin());

    size_t n = numMeshAxes();
    auto inputType = cast<RankedTensorType>(inputExpanded.getType());
    auto outputTileType = cast<RankedTensorType>(await.getValue().getType());
    ArrayRef<int64_t> inputTile = inputType.getShape().drop_front(n);
    ArrayRef<int64_t> outputTile = outputTileType.getShape();
    if (inputTile.size() != outputTile.size()) {
      fail(collective, "collective input and output tiles must have the same "
                       "rank");
      return;
    }

    CollectiveResolutionError resolutionError;
    FailureOr<CollectiveResolution> resolution = resolveCollectiveAtoms(
        collective, meshAxisTypes, inputTile, outputTile, resolutionError);
    if (failed(resolution)) {
      if (resolutionError.kind ==
          CollectiveResolutionError::Kind::NoCommonAtoms) {
        fail(collective,
             "distributed-lower-for-sanity-check could not split this "
             "collective's factors into a common set of atoms (" +
                 resolutionError.reasons.front() + ")");
      } else {
        for (const std::string &reason : resolutionError.reasons)
          fail(collective, reason);
      }
      return;
    }
    ArrayRef<ResolvedGroup> reductionGroups = resolution->reductionGroups;
    ArrayRef<std::pair<ResolvedGroup, ResolvedGroup>> pairs = resolution->pairs;
    const CollectiveAtoms &atoms = resolution->atoms;

    // 1. Expose every atom as its own dim.
    SmallVector<AtomLabel> labels;
    auto appendAxis = [&](AtomSpace space, size_t axisIdx) {
      labels.append(atoms.labelsOfAxis({space, axisIdx}));
    };
    for (size_t a = 0; a < n; ++a)
      appendAxis(AtomSpace::Mesh, a);
    for (size_t j = 0; j < inputTile.size(); ++j)
      appendAxis(AtomSpace::InTile, j);
    Value running = reshapeTo(inputExpanded, atoms.extentsOf(labels), loc);

    // 2. Reduce.
    for (auto [group, region] :
         llvm::zip_equal(reductionGroups, collective.getReductionBodies())) {
      auto elemType = cast<RankedTensorType>(running.getType()).getElementType();
      auto kind = stablehlo::classifyReduceBlockKind(region.front());
      Value identity = stablehlo::getIdentityValueForReduceKind(builder, loc,
                                                                elemType, kind);
      if (!identity) {
        fail(collective,
            "distributed-lower-for-sanity-check requires a collective's "
            "reduction body to be a single recognized associative op "
            "(add/mul/min/max/and/or/xor) with a known identity element over "
            "its element type");
        return;
      }
      for (AtomLabel label : atoms.labelsOf(group)) {
        auto dimPos = llvm::find(labels, label);
        if (label.space != AtomSpace::Mesh || dimPos == labels.end()) {
          fail(collective, "reduction over an axis that is not a live mesh "
                           "axis (or is reduced twice)");
          return;
        }
        size_t dim = dimPos - labels.begin();
        running = foldReduction(running, dim, region.front(), identity, loc);
        labels.erase(dimPos);
      }
    }

    // 3. Relabel from input atoms to output atoms. All lookups use the
    // pre-relabel labels so a permutation of atoms doesn't chase itself.
    SmallVector<AtomLabel> relabeled = labels;
    SmallVector<bool> dropped(labels.size(), false);
    SmallVector<AtomLabel> namedIn, namedOut;
    auto dimOf = [&](AtomLabel label) -> std::optional<size_t> {
      auto pos = llvm::find(labels, label);
      if (pos == labels.end())
        return std::nullopt;
      return pos - labels.begin();
    };
    for (const auto &[lhs, rhs] : pairs) {
      SmallVector<AtomLabel> lhsLabels = atoms.labelsOf(lhs);
      SmallVector<AtomLabel> rhsLabels = atoms.labelsOf(rhs);
      for (auto [l, r] : llvm::zip_equal(lhsLabels, rhsLabels)) {
        if (l.space != AtomSpace::Replicate)
          namedIn.push_back(l);
        if (r.space != AtomSpace::Replicate)
          namedOut.push_back(r);
        if (l.space == AtomSpace::Replicate)
          continue;
        std::optional<size_t> dim = dimOf(l);
        if (!dim) {
          fail(collective, "mapping lhs references an already-consumed axis");
          return;
        }
        if (r.space == AtomSpace::Replicate)
          dropped[*dim] = true;
        else
          relabeled[*dim] = r;
      }
    }
    // An input mesh atom no pair consumes normally passes through unchanged.
    // If some pair writes that same atom instead, the input was uniform along
    // it, so only its index-0 representative is kept and the write (or the
    // final broadcast) re-expands it.
    for (AtomLabel label : namedOut)
      if (label.space == AtomSpace::Mesh && !llvm::is_contained(namedIn, label))
        if (std::optional<size_t> dim = dimOf(label))
          dropped[*dim] = true;
    // The verifier requires the mapping to cover every tile dimension in full,
    // so an unnamed tile atom has extent 1: an input one is dropped, an
    // output one is filled by the final broadcast.
    for (size_t j = 0; j < inputTile.size(); ++j) {
      for (AtomLabel label : atoms.labelsOfAxis({AtomSpace::InTile, j})) {
        if (llvm::is_contained(namedIn, label))
          continue;
        assert(atoms.extentOf(label) == 1 &&
               "collective mapping must cover the whole input tile");
        dropped[*dimOf(label)] = true;
      }
    }
    for (size_t dim = labels.size(); dim-- > 0;) {
      if (!dropped[dim])
        continue;
      running = dropDim(running, dim, loc);
      relabeled.erase(relabeled.begin() + dim);
    }

    // The result varies across the mesh iff a mesh atom survives from the
    // input; every other mesh atom is a clone made by the final broadcast.
    bool varies = llvm::any_of(relabeled, [&](const AtomLabel &label) {
      return label.space == AtomSpace::Mesh && atoms.extentOf(label) > 1;
    });

    // 4. Place into [mesh atoms..., output tile atoms...] and reshape.
    SmallVector<AtomLabel> target;
    for (size_t a = 0; a < n; ++a)
      target.append(atoms.labelsOfAxis({AtomSpace::Mesh, a}));
    for (size_t j = 0; j < outputTile.size(); ++j)
      target.append(atoms.labelsOfAxis({AtomSpace::OutTile, j}));
    SmallVector<int64_t> targetPos;
    for (AtomLabel label : relabeled) {
      auto pos = llvm::find(target, label);
      if (pos == target.end() ||
          llvm::is_contained(targetPos, pos - target.begin())) {
        fail(collective, "collective mapping leaves an atom unplaced or "
                         "claims one output atom twice");
        return;
      }
      targetPos.push_back(pos - target.begin());
    }
    Value placed = placeAtoms(running, targetPos, atoms.extentsOf(target), loc);
    SmallVector<int64_t> finalShape(meshExtents);
    finalShape.append(outputTile.begin(), outputTile.end());
    Value finalExpanded = reshapeTo(placed, finalShape, loc);

    // Memoize the expanded form under the Await's result so a downstream
    // kernel/collective consuming it directly (no cast) resolves it via
    // getExpandedValue's memo lookup; additionally, if the Await's result is
    // itself immediately globalized by a real cast (e.g. feeding the
    // function's own return directly), collapse and map that cast's output
    // too, mirroring lowerKernel's handling of its own results.
    expandedOf[await.getValue()] = finalExpanded;
    if (varies)
      deviceVaryingResults.insert(await.getValue());
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

  Value reshapeTo(Value tensor, ArrayRef<int64_t> shape, Location loc) {
    auto type = cast<RankedTensorType>(tensor.getType());
    if (type.getShape() == shape)
      return tensor;
    return builder.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(shape, type.getElementType()), tensor);
  }

  // Transposes `tensor`'s dims into increasing `targetPos` order, then
  // broadcasts it into `targetShape` (one dim per output atom), so an atom
  // with no source dim is uniform along it.
  Value placeAtoms(Value tensor, ArrayRef<int64_t> targetPos,
                   ArrayRef<int64_t> targetShape, Location loc) {
    auto type = cast<RankedTensorType>(tensor.getType());
    SmallVector<int64_t> order(targetPos.size());
    std::iota(order.begin(), order.end(), 0);
    llvm::sort(order, [&](int64_t a, int64_t b) {
      return targetPos[a] < targetPos[b];
    });
    if (!llvm::is_sorted(order)) {
      SmallVector<int64_t> shape;
      for (int64_t src : order)
        shape.push_back(type.getDimSize(src));
      tensor = builder.create<stablehlo::TransposeOp>(
          loc, RankedTensorType::get(shape, type.getElementType()), tensor,
          order);
    }
    SmallVector<int64_t> broadcastDims;
    for (int64_t src : order)
      broadcastDims.push_back(targetPos[src]);
    if (broadcastDims.size() == targetShape.size())
      return tensor;
    return builder.create<stablehlo::BroadcastInDimOp>(
        loc, RankedTensorType::get(targetShape, type.getElementType()), tensor,
        broadcastDims);
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
    Value sliced = cast<RankedTensorType>(tensor.getType()).getDimSize(dim) == 1
                       ? tensor
                       : sliceAtIndex(tensor, dim, 0, loc);
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
// Removes `sdy.sharding` from per-argument/result attribute dictionaries.
// Shardy's mesh declaration is already gone by this point, and a func.func
// would fail verification on a sharding naming it.
ArrayAttr stripSdyShardings(ArrayAttr attrs, Builder &builder) {
  if (!attrs)
    return attrs;
  SmallVector<Attribute> stripped;
  for (Attribute entry : attrs) {
    NamedAttrList dict(cast<DictionaryAttr>(entry));
    dict.erase("sdy.sharding");
    stripped.push_back(dict.getDictionary(builder.getContext()));
  }
  return builder.getArrayAttr(stripped);
}

// A distributed.function may use values defined at module scope (constants
// the export leaves there), but func.func is isolated from above. Clones each
// such value's defining op (and, transitively, what it depends on) to the
// start of the function.
void cloneCapturedValuesIntoFunc(func::FuncOp funcOp, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  IRMapping cloned;
  std::function<Value(Value)> capture = [&](Value value) -> Value {
    if (Value existing = cloned.lookupOrNull(value))
      return existing;
    Operation *def = value.getDefiningOp();
    assert(def && "only op results can be defined outside a function");
    for (Value operand : def->getOperands())
      capture(operand);
    builder.clone(*def, cloned);
    return cloned.lookup(value);
  };
  funcOp.walk([&](Operation *op) {
    for (OpOperand &use : op->getOpOperands()) {
      Operation *def = use.get().getDefiningOp();
      if (def && !funcOp->isAncestor(def))
        use.set(capture(use.get()));
    }
  });
}

func::FuncOp convertDistributedFunctionToFunc(DistributedFunctionOp distFn,
                                              OpBuilder &builder) {
  builder.setInsertionPoint(distFn);
  auto funcOp = builder.create<func::FuncOp>(
      distFn.getLoc(), distFn.getSymName(), distFn.getFunctionType(),
      distFn.getSymVisibilityAttr(),
      stripSdyShardings(distFn.getArgAttrsAttr(), builder),
      stripSdyShardings(distFn.getResAttrsAttr(), builder));
  funcOp.getBody().takeBody(distFn.getBody());
  auto &block = funcOp.getBody().front();
  cloneCapturedValuesIntoFunc(funcOp, builder);
  auto yieldOp = cast<DistributedYieldOp>(block.getTerminator());
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(yieldOp);
  builder.create<func::ReturnOp>(yieldOp.getLoc(), yieldOp.getReturns());
  yieldOp.erase();
  distFn.erase();
  return funcOp;
}

// Splices `call`'s callee body in at the call site, entirely avoiding the
// callee's own GLOBAL-scope argument/return values: a partitioned tensor
// argument/return of a DistributedFunctionOp is bound through exactly one
// CastGlobalToLocal/CastLocalToGlobal (see convertFunctionToDistributedFunction),
// whose local-scope side is already at exactly the same axes this call's own
// (also local-scope) operands/results share with the callee (DistributedCallOp
// reuses the callee's own partitioning_axes/argument_shardings/output_shardings
// directly, see localizeCall in ConvertMainToDistributedFunction.cpp); an
// unpartitioned one has no cast at all, local and global being identical. So
// rather than reconstituting a global value only to immediately re-derive its
// local form again, this skips cloning a boundary cast where one exists and
// wires the call's own local operand/result straight onto the value it would
// have produced/consumed -- exactly the flat program a hand-inlined version
// of the same call would produce. Leaves the callee function itself in
// place, since this pass only ever extracts `main`; it's a dead symbol at
// that point, cleaned up by the same subsequent DCE this pass already relies
// on for its axis.* metadata leftovers.
LogicalResult inlineDistributedCall(DistributedCallOp call) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<DistributedFunctionOp>(
      call, call.getCalleeAttr());
  Block &calleeBlock = callee.getBody().front();
  OpBuilder builder(call);
  IRMapping mapping;
  // Both boundary casts get skipped below rather than cloned, so a
  // partitioned result's producing kernel/collective keeps exactly the one
  // real consumer this pass's own bounding-cast lookup
  // (findBoundingCastLocalToGlobal) requires -- cloning the callee's own
  // result cast alongside redirecting the call's result to its input would
  // otherwise leave that value with two consumers.
  llvm::SmallDenseSet<Operation *> skip;

  for (auto [idx, arg] : llvm::enumerate(calleeBlock.getArguments())) {
    Value operand = call.getArguments()[idx];
    // An unpartitioned argument is identical at both scopes (no data to
    // reconcile), so the callee never needs a cast for it at all -- map it
    // straight through regardless of whether the callee body happens to
    // wrap it in a (then-trivial) cast anyway.
    if (arg.getType() == operand.getType()) {
      mapping.map(arg, operand);
      continue;
    }
    auto castOp = arg.hasOneUse()
                      ? dyn_cast<DistributedCastGlobalToLocalOp>(
                            *arg.getUsers().begin())
                      : nullptr;
    if (!castOp)
      return call.emitError()
             << "argument " << idx
             << ": expected exactly one direct CastGlobalToLocal off this "
                "callee argument to inline this call";
    mapping.map(castOp.getOutput(), operand);
    skip.insert(castOp);
  }

  auto yieldOp = cast<DistributedYieldOp>(calleeBlock.getTerminator());
  SmallVector<Value> resultSources;
  for (auto [idx, result] : llvm::enumerate(call.getResults())) {
    Value returned = yieldOp.getReturns()[idx];
    if (result.getType() == returned.getType()) {
      resultSources.push_back(returned);
      continue;
    }
    auto castOp = returned.getDefiningOp<DistributedCastLocalToGlobalOp>();
    if (!castOp)
      return call.emitError()
             << "result " << idx
             << ": expected a direct CastLocalToGlobal producing this "
                "callee return to inline this call";
    skip.insert(castOp);
    resultSources.push_back(castOp.getInput());
  }

  for (Operation &op : calleeBlock.without_terminator())
    if (!skip.contains(&op))
      builder.clone(op, mapping);

  for (auto [result, source] : llvm::zip_equal(call.getResults(), resultSources))
    result.replaceAllUsesWith(mapping.lookupOrDefault(source));
  call.erase();
  return success();
}

// Inlines every distributed.DistributedCall reachable from `mainFn`, so the
// rest of this pass can keep treating "main's body" as the whole program,
// exactly as it would if the source had never been factored into functions.
// The call DAG is non-recursive, so repeatedly inlining whatever call is
// found first terminates: a callee's own calls are cloned in as fresh ops
// and picked up by the next scan.
LogicalResult inlineDistributedCalls(DistributedFunctionOp mainFn) {
  Block &block = mainFn.getBody().front();
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto call = dyn_cast<DistributedCallOp>(&op)) {
        if (failed(inlineDistributedCall(call)))
          return failure();
        changed = true;
      }
    }
  }
  return success();
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
    if (failed(inlineDistributedCalls(distFn))) {
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

    SmallVector<Operation *> originals;
    for (Operation &op : mainBlock.without_terminator())
      originals.push_back(&op);

    SanityCheckLowering lowering(builder, meshAxisTypes);
    lowering.lowerBlock(mainBlock);
    if (lowering.sawFailure()) {
      signalPassFailure();
      return;
    }

    auto yieldOp = cast<DistributedYieldOp>(mainBlock.getTerminator());
    SmallVector<Value> newReturns;
    for (Value v : yieldOp.getReturns())
      newReturns.push_back(lowering.getFlatValue(v));
    builder.setInsertionPoint(yieldOp);
    builder.create<DistributedYieldOp>(yieldOp.getLoc(), newReturns);
    yieldOp.erase();

    // Everything was re-expressed as newly cloned plain stablehlo, so erase
    // every original op, in reverse so a value's uses go before its def.
    for (Operation *op : llvm::reverse(originals)) {
      op->dropAllUses();
      op->erase();
    }

    convertDistributedFunctionToFunc(distFn, builder);

    // The output is plain func/stablehlo: drop the physical mesh, axis
    // algebra, and any callee distributed.function that inlining left dead.
    // Functions go first since they are the only remaining users of the
    // module-scope axis values.
    SmallVector<Operation *> metadata;
    for (Operation &op :
         llvm::make_early_inc_range(module.getBody()->getOperations())) {
      if (isa<DistributedFunctionOp>(op))
        op.erase();
      else if (llvm::is_contained({"axis", "distributed"},
                                  op.getName().getDialectNamespace()))
        metadata.push_back(&op);
    }
    for (Operation *op : llvm::reverse(metadata)) {
      op->dropAllUses();
      op->erase();
    }
    // Module-scope constants are now redundant with the copies cloned into
    // the function.
    for (Operation &op :
         llvm::make_early_inc_range(llvm::reverse(*module.getBody())))
      if (!isa<SymbolOpInterface>(op) && isOpTriviallyDead(&op))
        op.erase();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
