#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/FindShardyFunctionsAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <numeric>
#include <optional>
#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Casts a range of values/typed-values to untyped values, mostly for
// meeting builder interface requirements.
template <typename RangeT>
static llvm::SmallVector<Value> asValues(const RangeT &groups) {
  llvm::SmallVector<Value> values;
  values.reserve(groups.size());
  for (Value group : groups) {
    values.push_back(group);
  }
  return values;
}

// Rewrite pattern to convert shardy functions (func.func with shardy ops)
// to distributed functions within a given mesh computations.
struct FuncToDistributedFunctionPattern
    : public OpRewritePattern<func::FuncOp> {
  FuncToDistributedFunctionPattern(
      MLIRContext *context, TypedValue<axis::FactorGroupType> contextGroup)
      : OpRewritePattern<func::FuncOp>(context), contextGroup(contextGroup) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (!funcOp->getParentOfType<distributed::MeshComputationOp>()) {
      return failure();
    }

    SmallVector<Attribute> argumentTypes;
    argumentTypes.reserve(funcOp.getFunctionType().getNumInputs());
    for (Type inputType : funcOp.getFunctionType().getInputs()) {
      argumentTypes.push_back(TypeAttr::get(inputType));
    }

    SmallVector<Attribute> returnTypes;
    returnTypes.reserve(funcOp.getFunctionType().getNumResults());
    for (Type resultType : funcOp.getFunctionType().getResults()) {
      returnTypes.push_back(TypeAttr::get(resultType));
    }

    auto distributedFunc = rewriter.create<distributed::DistributedFunctionOp>(
        funcOp.getLoc(), funcOp.getSymNameAttr(), contextGroup,
        rewriter.getArrayAttr(argumentTypes),
        rewriter.getArrayAttr(returnTypes));

    for (NamedAttribute attr : funcOp->getAttrs()) {
      if (attr.getName() == rewriter.getStringAttr("sym_name") ||
          attr.getName() == rewriter.getStringAttr("function_type")) {
        continue;
      }
      distributedFunc->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), distributedFunc.getBody(),
                                distributedFunc.getBody().end());
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  TypedValue<axis::FactorGroupType> contextGroup;
};

struct FuncCallToDistributedCallPattern
    : public OpRewritePattern<func::CallOp> {
  FuncCallToDistributedCallPattern(
      MLIRContext *context,
      const llvm::DenseSet<StringAttr> &distributedFunctionNames,
      TypedValue<axis::FactorGroupType> replicateOver)
      : OpRewritePattern<func::CallOp>(context),
        distributedFunctionNames(distributedFunctionNames),
        replicateOver(replicateOver) {}

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (!callOp->getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    StringRef callee = callOp.getCallee();
    if (!distributedFunctionNames.contains(rewriter.getStringAttr(callee))) {
      return failure();
    }

    auto distributedCall = rewriter.create<distributed::DistributedCallOp>(
        callOp.getLoc(), callOp.getResultTypes(), callOp.getCalleeAttr(),
        replicateOver, callOp.getOperands());
    rewriter.replaceOp(callOp, distributedCall->getResults());
    return success();
  }

private:
  llvm::DenseSet<StringAttr> distributedFunctionNames;
  TypedValue<axis::FactorGroupType> replicateOver;
};

struct FuncReturnToDistributedYieldPattern
    : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    if (!returnOp->getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    rewriter.create<distributed::DistributedYieldOp>(returnOp.getLoc(),
                                                     returnOp.getOperands());
    rewriter.eraseOp(returnOp);
    return success();
  }
};

// Logic implemented per collective op to extract the
// reduction and permuted dimension mappings.
template <typename StablehloCollectiveOp>
LogicalResult getDimMappings(
    StablehloCollectiveOp op, PatternRewriter &rewriter,
    const llvm::StringMap<Value> &shardyToDistributedAxis,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> meshAxisFactorsMajorFirst,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> inputTensorFactors,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> outputTensorFactors,
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> &out_reductionDims,
    std::optional<FlatSymbolRefAttr> &out_reductionFunction,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingLHS,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingRHS);

// Parses replica_group_mesh_axes and resolves each named shardy axis to a
// distributed axis factor.
template <typename StablehloCollectiveOp>
LogicalResult collectReplicaGroupMeshAxes(
    StablehloCollectiveOp op,
    const llvm::StringMap<Value> &shardyToDistributedAxis,
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> &out_meshAxes) {
  auto replicaGroupsAttr = op.getReplicaGroups();
  using ReplicaGroupMeshAxesAttr = ::mlir::stablehlo::ReplicaGroupMeshAxesAttr;
  ReplicaGroupMeshAxesAttr replicaGroupMeshAxesAttr =
      dyn_cast<ReplicaGroupMeshAxesAttr>(replicaGroupsAttr);
  if (!replicaGroupMeshAxesAttr) {
    return op.emitOpError(
               "expected op to have #stablehlo.replica_group_mesh_axes "
               "attribute, instead found ")
           << replicaGroupsAttr;
  }

  for (Attribute axisAttr : replicaGroupMeshAxesAttr.getAxes()) {
    std::string axisName;
    if (auto axisNameAttr = dyn_cast<StringAttr>(axisAttr)) {
      axisName = axisNameAttr.getValue().str();
    } else if (auto axisRefAttr = dyn_cast<sdy::AxisRefAttr>(axisAttr)) {
      axisName = axisRefAttr.getName().str();
    }
    if (axisName.empty()) {
      return op.emitOpError(
                 "expected replica_group_mesh_axes to contain sdy.axis_ref "
                 "or string attributes, instead found ")
             << axisAttr;
    }

    auto axisMappingIt = shardyToDistributedAxis.find(axisName);
    if (axisMappingIt == shardyToDistributedAxis.end()) {
      return op.emitOpError(
                 "expected replica_group_mesh_axes axis to have a mapping "
                 "in shardyToDistributedAxis, instead found none for axis ")
             << axisName;
    }

    out_meshAxes.push_back(castTypedValue<axis::AxisFactorType>(
        axisMappingIt->getValue(), "AxisFactorType"));
  }

  return success();
}

// Specialization for stablehlo.all_gather
template <>
LogicalResult getDimMappings(
    stablehlo::AllGatherOp op, PatternRewriter &rewriter,
    const llvm::StringMap<Value> &shardyToDistributedAxis,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> meshAxisFactorsMajorFirst,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> inputTensorFactors,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> outputTensorFactors,
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> &out_reductionDims,
    std::optional<FlatSymbolRefAttr> &out_reductionFunction,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingLHS,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingRHS) {
  // Assert single-tensor operand
  if (op->getNumOperands() != 1) {
    return op.emitOpError(
        "expected all_gather to have a single tensor operand");
  }

  // For this collective, no reduction dims, but we need to know which part
  // of the output tensor to map our gather axis to: gatherOn --> newTensorAx,
  // and replicate(N) --> gatherOn. The allGather op gives a tensor rank to
  // concat on, so we need to take the R rank axis from the output tensor and
  // factor it into two parts: the most major part with extent equal number of
  // devices, and the minor part which we expect to have extent equal to the
  // input tensor's axis extent. Then we map all of our mesh axis to the major
  // part. Assume that majority of concatenation follows the majority in the
  // axis attribute from the shardy op.

  llvm::SmallVector<TypedValue<axis::AxisFactorType>> gatheredMeshAxes;
  if (failed(collectReplicaGroupMeshAxes(op, shardyToDistributedAxis,
                                         gatheredMeshAxes))) {
    return failure();
  }

  int64_t concatDim = op.getAllGatherDim();
  if (concatDim < 0 ||
      concatDim >= static_cast<int64_t>(inputTensorFactors.size()) ||
      concatDim >= static_cast<int64_t>(outputTensorFactors.size())) {
    return op.emitOpError() << "all_gather dimension " << concatDim
                            << " out of range for tensor ranks (input "
                            << inputTensorFactors.size() << ", output "
                            << outputTensorFactors.size() << ")";
  }

  auto outputGatherFactor = outputTensorFactors[concatDim];
  auto outputGatherAxis = axis::getFactorProvenanceAxis(outputGatherFactor);
  if (failed(outputGatherAxis)) {
    return op.emitOpError(
        "failed to resolve provenance axis for all_gather output factor");
  }

  // The size of the concatenated output dimension should be equal to the
  // product of the gather-over dimensions and the corresponding rank of the
  // input tensor.
  llvm::SmallVector<int32_t> gatherRefactorExtents;
  gatherRefactorExtents.reserve(gatheredMeshAxes.size() + 1);
  for (TypedValue<axis::AxisFactorType> gatheredMeshAxis : gatheredMeshAxes) {
    gatherRefactorExtents.push_back(
        static_cast<int32_t>(gatheredMeshAxis.getType().getExtent()));
  }
  gatherRefactorExtents.push_back(static_cast<int32_t>(
      inputTensorFactors[concatDim].getType().getExtent()));

  llvm::SmallVector<TypedValue<axis::AxisFactorType>>
      gatheredOutputFactorValues = axis::factorAxisByExtents(
          *outputGatherAxis, gatherRefactorExtents, rewriter, op.getLoc());

  // map each gathered mesh axis to its matching major output gather factor.
  for (auto [idx, gatheredMeshAxis] : llvm::enumerate(gatheredMeshAxes)) {
    out_mappingLHS.push_back({gatheredMeshAxis});
    out_mappingRHS.push_back({gatheredOutputFactorValues[idx]});
  }

  // map the input gather dimension to the remaining minor output gather factor.
  out_mappingLHS.push_back({inputTensorFactors[concatDim]});
  out_mappingRHS.push_back({gatheredOutputFactorValues.back()});

  // Identity map remaining tensor factors.
  for (int64_t i = 0; i < static_cast<int64_t>(inputTensorFactors.size());
       ++i) {
    if (i == concatDim) {
      continue;
    }
    auto inputFactor = inputTensorFactors[i];
    auto outputFactor = outputTensorFactors[i];
    // input, output tensor types not the same so we only expect
    // matching extents.
    auto in_extent = inputFactor.getType().getExtent();
    auto out_extent = outputFactor.getType().getExtent();
    if (in_extent != out_extent) {
      return op.emitOpError(
                 "expected non-gather input/output tensor factors to have "
                 "equal index space, instead found ")
             << inputFactor << " and " << outputFactor;
    }
    out_mappingLHS.push_back({inputFactor});
    out_mappingRHS.push_back({outputFactor});
  }

  // Identity map mesh axes that are not part of this gather.
  for (TypedValue<axis::AxisFactorType> axisFactor :
       meshAxisFactorsMajorFirst) {
    if (std::find(gatheredMeshAxes.begin(), gatheredMeshAxes.end(),
                  axisFactor) == gatheredMeshAxes.end()) {
      out_mappingLHS.push_back({axisFactor});
      out_mappingRHS.push_back({axisFactor});
    }
  }

  // all_gather has no reductions.
  out_reductionDims.clear();
  out_reductionFunction = std::nullopt;
  return success();
}

// Specialization for collective permute
template <>
LogicalResult getDimMappings(
    stablehlo::CollectivePermuteOp op, PatternRewriter &rewriter,
    const llvm::StringMap<Value> &shardyToDistributedAxis,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> meshAxisFactorsMajorFirst,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> inputTensorFactors,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> outputTensorFactors,
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> &out_reductionDims,
    std::optional<FlatSymbolRefAttr> &out_reductionFunction,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingLHS,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingRHS) {
  // Assert single-tensor result.
  if (op->getNumResults() != 1) {
    return op.emitOpError(
        "expected collective_permute to have a single tensor result");
  }
  // Unfortunately collective-permute is specified in terms of source-target
  // pairs. We must infer the dimension mapping from looking at the indices.
  // There are two cases of an all-permute:
  // 1. The permutation is a simple reordering of the axes within the same
  //    tensor rank.
  // 2. Axes swap between tensor ranks, but the axis extents match. (Otherwise,
  //    tile shape or content changes, which is not a simple permute.)
  // Fortunately I think we can handle these the same way. If we assume that
  // the source/target pairs are generated by some permutation of axes ordering,
  // then if an axis has stride N in the old majority ordering, and the mapping
  // contains [N --> M], then I think it maps to stride M in the new majority
  // ordering. So then we can check each element of the source/target pairs and
  // check that (in % N == out % M), and if this is true for each pair and each
  // factor then our mapping holds. Then we just need to sort the mesh axis by
  // their stride to get their positions in the new ordering, and map one giant
  // group of mesh factors to mesh factors (since without matching size
  // gaurantees we don't know if the mapping is divisible)

  (void)rewriter;

  llvm::SmallVector<TypedValue<axis::AxisFactorType>> orderedMeshFactors(
      meshAxisFactorsMajorFirst.begin(), meshAxisFactorsMajorFirst.end());

  if (orderedMeshFactors.empty()) {
    return op.emitOpError(
        "expected at least one mesh axis factor for collective_permute");
  }

  auto sourceTargetPairsAttr = dyn_cast_or_null<DenseIntElementsAttr>(
      op->getAttr("source_target_pairs"));
  if (!sourceTargetPairsAttr) {
    return op.emitOpError(
        "expected dense integer source_target_pairs attribute");
  }
  auto sourceTargetPairsType =
      dyn_cast<RankedTensorType>(sourceTargetPairsAttr.getType());
  if (!sourceTargetPairsType || sourceTargetPairsType.getRank() != 2 ||
      sourceTargetPairsType.getDimSize(1) != 2) {
    return op.emitOpError(
        "expected source_target_pairs type tensor<Nx2xinteger>");
  }

  llvm::SmallVector<int64_t> sourceTargetPairs;
  sourceTargetPairs.reserve(sourceTargetPairsAttr.getNumElements());
  for (APInt value : sourceTargetPairsAttr.getValues<APInt>()) {
    sourceTargetPairs.push_back(value.getSExtValue());
  }
  if (sourceTargetPairs.empty() || (sourceTargetPairs.size() % 2) != 0) {
    return op.emitOpError(
        "expected non-empty source_target_pairs with [source, target] rows");
  }

  llvm::SmallVector<uint64_t> extents;
  extents.reserve(orderedMeshFactors.size());
  for (TypedValue<axis::AxisFactorType> factor : orderedMeshFactors) {
    extents.push_back(static_cast<uint64_t>(factor.getType().getExtent()));
  }

  auto buildMajorToMinorStrides = [](llvm::ArrayRef<uint64_t> axisExtents) {
    llvm::SmallVector<uint64_t> strides(axisExtents.size(), 1);
    uint64_t running = 1;
    for (int i = static_cast<int>(axisExtents.size()) - 1; i >= 0; --i) {
      strides[i] = running;
      running *= axisExtents[i];
    }
    return strides;
  };

  uint64_t totalMeshSize = 1;
  for (uint64_t extent : extents) {
    totalMeshSize *= extent;
  }
  for (size_t i = 0; i < sourceTargetPairs.size(); i += 2) {
    int64_t source = sourceTargetPairs[i];
    int64_t target = sourceTargetPairs[i + 1];
    if (source < 0 || target < 0 ||
        static_cast<uint64_t>(source) >= totalMeshSize ||
        static_cast<uint64_t>(target) >= totalMeshSize) {
      return op.emitOpError()
             << "source_target_pairs contains id outside mesh range [0, "
             << totalMeshSize << "): [" << source << ", " << target << "]";
    }
  }

  llvm::SmallVector<uint64_t> sourceStrides = buildMajorToMinorStrides(extents);

  llvm::SmallVector<unsigned> permutation(extents.size());
  std::iota(permutation.begin(), permutation.end(), 0);

  llvm::SmallVector<unsigned> inferredPermutation;
  bool foundPermutation = false;
  bool ambiguousPermutation = false;

  do {
    llvm::SmallVector<uint64_t> permutedExtents;
    permutedExtents.reserve(extents.size());
    for (unsigned axisIdx : permutation) {
      permutedExtents.push_back(extents[axisIdx]);
    }

    llvm::SmallVector<uint64_t> targetOrderStrides =
        buildMajorToMinorStrides(permutedExtents);
    llvm::SmallVector<uint64_t> targetStridesByAxis(extents.size(), 0);
    for (auto [position, axisIdx] : llvm::enumerate(permutation)) {
      targetStridesByAxis[axisIdx] = targetOrderStrides[position];
    }

    bool permutationMatches = true;
    for (size_t i = 0; i < sourceTargetPairs.size() && permutationMatches;
         i += 2) {
      uint64_t source = static_cast<uint64_t>(sourceTargetPairs[i]);
      uint64_t target = static_cast<uint64_t>(sourceTargetPairs[i + 1]);
      for (size_t axisIdx = 0; axisIdx < extents.size(); ++axisIdx) {
        uint64_t sourceCoord =
            (source / sourceStrides[axisIdx]) % extents[axisIdx];
        uint64_t targetCoord =
            (target / targetStridesByAxis[axisIdx]) % extents[axisIdx];
        if (sourceCoord != targetCoord) {
          permutationMatches = false;
          break;
        }
      }
    }

    if (permutationMatches) {
      if (foundPermutation) {
        ambiguousPermutation = true;
        break;
      }
      inferredPermutation = permutation;
      foundPermutation = true;
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  if (!foundPermutation) {
    return op.emitOpError(
        "failed to infer mesh-axis permutation from source_target_pairs");
  }
  if (ambiguousPermutation) {
    return op.emitOpError(
        "source_target_pairs underconstrained: multiple mesh permutations "
        "satisfy the mapping");
  }

  llvm::SmallVector<TypedValue<axis::AxisFactorType>> sourceMeshFactors;
  llvm::SmallVector<TypedValue<axis::AxisFactorType>> targetMeshFactors;
  sourceMeshFactors.reserve(orderedMeshFactors.size());
  targetMeshFactors.reserve(orderedMeshFactors.size());
  sourceMeshFactors.append(orderedMeshFactors.begin(),
                           orderedMeshFactors.end());
  for (unsigned axisIdx : inferredPermutation) {
    targetMeshFactors.push_back(orderedMeshFactors[axisIdx]);
  }
  out_mappingLHS.push_back(sourceMeshFactors);
  out_mappingRHS.push_back(targetMeshFactors);

  // collective_permute should preserve local tensor shape; map tensor factors
  // by rank position with equal extent.
  if (inputTensorFactors.size() != outputTensorFactors.size()) {
    return op.emitOpError(
        "expected input/output tensor ranks to match for collective_permute");
  }
  for (size_t i = 0; i < inputTensorFactors.size(); ++i) {
    auto inputFactor = inputTensorFactors[i];
    auto outputFactor = outputTensorFactors[i];
    if (inputFactor.getType().getExtent() !=
        outputFactor.getType().getExtent()) {
      return op.emitOpError(
                 "expected collective_permute to preserve tensor dim extents, "
                 "instead found ")
             << inputFactor << " and " << outputFactor;
    }
    out_mappingLHS.push_back({inputFactor});
    out_mappingRHS.push_back({outputFactor});
  }

  out_reductionDims.clear();
  out_reductionFunction = std::nullopt;
  return success();
}

// Specialization for stablehlo.all_reduce
template <>
LogicalResult getDimMappings(
    stablehlo::AllReduceOp op, PatternRewriter &rewriter,
    const llvm::StringMap<Value> &shardyToDistributedAxis,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> meshAxisFactorsMajorFirst,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> inputTensorFactors,
    llvm::ArrayRef<TypedValue<axis::AxisFactorType>> outputTensorFactors,
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> &out_reductionDims,
    std::optional<FlatSymbolRefAttr> &out_reductionFunction,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingLHS,
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        &out_mappingRHS) {
  // Assert single-tensor operand
  if (op.getOperands().size() != 1) {
    return op.emitOpError(
        "expected all_reduce to have a single tensor operand");
  }

  llvm::SmallVector<TypedValue<axis::AxisFactorType>> reductionDims;
  if (failed(collectReplicaGroupMeshAxes(op, shardyToDistributedAxis,
                                         reductionDims))) {
    return failure();
  }

  for (TypedValue<axis::AxisFactorType> reducedFactor : reductionDims) {
    // Since this is an all-reduce, every device in that spatial dimension
    // gets the same value, so we need to add replicate(n) --> reduceAx mapping
    int extent = static_cast<int>(reducedFactor.getType().getExtent());
    auto replicationAxis = rewriter.create<distributed::ReplicationAxisOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(extent));
    auto replicationFactors = axis::viewAxesAsFactors(
        ValueRange{replicationAxis.getAxis()}, rewriter, op.getLoc());
    if (replicationFactors.size() != 1) {
      return op.emitOpError(
          "failed to create replication factor for all_reduce mapping");
    }
    auto replicationFactor = replicationFactors.front();
    out_mappingLHS.push_back({replicationFactor});
    out_mappingRHS.push_back({reducedFactor});
  }

  // Need to identity map the mesh, tensor axis not being reduced.
  assert(inputTensorFactors.size() == outputTensorFactors.size() &&
         "expected input and output tensor factors to have the same size");
  for (int i = 0; i < inputTensorFactors.size(); ++i) {
    auto inputFactor = inputTensorFactors[i];
    auto outputFactor = outputTensorFactors[i];
    if (axis::areFactorIndexSpacesEqual({inputFactor}, {outputFactor}) ==
        false) {
      return op.emitOpError(
                 "expected input and output tensor factors to be identical, "
                 "instead found ")
             << inputFactor << " and " << outputFactor;
    }
    // add the axis factor to the mapping lists.
    out_mappingLHS.push_back({inputFactor});
    out_mappingRHS.push_back({outputFactor});
  }
  // Anything left in the map that isn't in the reductionDims should also be
  // identity mapped.
  for (TypedValue<axis::AxisFactorType> axisFactor :
       meshAxisFactorsMajorFirst) {
    if (std::find(reductionDims.begin(), reductionDims.end(), axisFactor) ==
        reductionDims.end()) {
      out_mappingLHS.push_back({axisFactor});
      out_mappingRHS.push_back({axisFactor});
    }
  }

  // TODO we need to make reduction function somewhere,
  // for now make an obviously bogus symbol
  out_reductionDims = reductionDims;
  out_reductionFunction = FlatSymbolRefAttr::get(
      op.getContext(),
      "not_yet_implemented_sdy_to_distributed_all_reduce_conversion");

  return success();
}

template <typename StablehloCollectiveOp>
struct StablehloCollectiveToDistributedCollectivePattern
    : public OpRewritePattern<StablehloCollectiveOp> {
  StablehloCollectiveToDistributedCollectivePattern(
      MLIRContext *context,
      const llvm::StringMap<Value> &shardyToDistributedAxis,
      llvm::ArrayRef<TypedValue<axis::AxisFactorType>>
          meshAxisFactorsMajorFirst)
      : OpRewritePattern<StablehloCollectiveOp>(context),
        shardyToDistributedAxis(shardyToDistributedAxis),
        meshAxisFactorsMajorFirst(meshAxisFactorsMajorFirst.begin(),
                                  meshAxisFactorsMajorFirst.end()) {}

  LogicalResult matchAndRewrite(StablehloCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->template getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected collective with one input and one output");
    }

    auto inputTensorType = dyn_cast<ShapedType>(op->getOperand(0).getType());
    if (!inputTensorType || !inputTensorType.hasRank()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked shaped input type");
    }

    auto outputTensorType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!outputTensorType || !outputTensorType.hasRank()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked shaped result type");
    }

    llvm::SmallVector<TypedValue<axis::AxisTypeInterface>> inputTensorAxes =
        axis::createAxesForRankedShape(inputTensorType, rewriter, op.getLoc());
    llvm::SmallVector<TypedValue<axis::AxisTypeInterface>> outputTensorAxes =
        axis::createAxesForRankedShape(outputTensorType, rewriter, op.getLoc());
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> inputTensorFactors =
        axis::viewAxesAsFactors(inputTensorAxes, rewriter, op.getLoc());
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> outputTensorFactors =
        axis::viewAxesAsFactors(outputTensorAxes, rewriter, op.getLoc());

    // Get the mappings per collective op.
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> reductionDims;
    std::optional<FlatSymbolRefAttr> reductionFunction;
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        mappingLHS;
    llvm::SmallVector<llvm::SmallVector<TypedValue<axis::AxisFactorType>>>
        mappingRHS;
    auto result = getDimMappings(op, rewriter, shardyToDistributedAxis,
                                 meshAxisFactorsMajorFirst, inputTensorFactors,
                                 outputTensorFactors, reductionDims,
                                 reductionFunction, mappingLHS, mappingRHS);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension mappings for collective op");
    }

    // otherwise, assemble the lists into a true distributed collective op, and
    // replace the original stablehlo collective op with it + the distributed
    // await op.
    // Step 1: need ot take our vectors of factors into factor group values.
    auto makeFactorGroup =
        [&](const llvm::SmallVector<TypedValue<axis::AxisFactorType>> &factors)
        -> TypedValue<axis::FactorGroupType> {
      llvm::SmallVector<Value> factorValues = asValues(factors);
      return rewriter
          .create<axis::AxisProductOp>(op.getLoc(), ValueRange(factorValues))
          .getProduct();
    };
    auto makeFactorGroupList =
        [&](const llvm::SmallVector<
            llvm::SmallVector<TypedValue<axis::AxisFactorType>>> &factorLists)
        -> llvm::SmallVector<TypedValue<axis::FactorGroupType>> {
      llvm::SmallVector<TypedValue<axis::FactorGroupType>> factorGroups;
      factorGroups.reserve(factorLists.size());
      for (const auto &factors : factorLists) {
        factorGroups.push_back(makeFactorGroup(factors));
      }
      return factorGroups;
    };

    // HLO collectives lower to a single reduction group that may contain
    // zero or more factors.
    llvm::SmallVector<TypedValue<axis::FactorGroupType>> reductionGroups;
    reductionGroups.reserve(1);
    llvm::SmallVector<Attribute> reductionFunctions;
    reductionFunctions.reserve(reductionFunction.has_value() ? 1 : 0);

    reductionGroups.push_back(makeFactorGroup(reductionDims));
    if (reductionFunction.has_value()) {
      reductionFunctions.push_back(*reductionFunction);
    }

    llvm::SmallVector<TypedValue<axis::FactorGroupType>> mappingLHSGroups =
        makeFactorGroupList(mappingLHS);
    llvm::SmallVector<TypedValue<axis::FactorGroupType>> mappingRHSGroups =
        makeFactorGroupList(mappingRHS);

    llvm::SmallVector<Value> reductionGroupValues = asValues(reductionGroups);
    llvm::SmallVector<Value> mappingLHSValues = asValues(mappingLHSGroups);
    llvm::SmallVector<Value> mappingRHSValues = asValues(mappingRHSGroups);

    // Step 2: create the distributed collective op.
    auto enclosingDistributedFunction =
        op->template getParentOfType<distributed::DistributedFunctionOp>();
    auto executionContext = enclosingDistributedFunction.getExecutionContext();

    auto outputTensorAsTensorType = dyn_cast<TensorType>(outputTensorType);
    if (!outputTensorAsTensorType) {
      return rewriter.notifyMatchFailure(op, "expected tensor result type");
    }

    auto distributedCollective =
        rewriter.create<distributed::DistributedCollectiveOp>(
            op.getLoc(), op->getOperand(0), executionContext, executionContext,
            ValueRange(reductionGroupValues),
            rewriter.getArrayAttr(reductionFunctions),
            ValueRange(mappingLHSValues), ValueRange(mappingRHSValues),
            TypeAttr::get(outputTensorAsTensorType));

    // Step 3: get the distributed collective op result handle,
    // and create a distributed await op to replace the original stablehlo
    // collective op.
    auto distributedAwait = rewriter.create<distributed::DistributedAwait>(
        op.getLoc(), outputTensorAsTensorType,
        distributedCollective.getAsyncHandle());
    rewriter.replaceOp(op, distributedAwait.getValue());
    return success();
  }

private:
  llvm::StringMap<Value> shardyToDistributedAxis;
  llvm::SmallVector<TypedValue<axis::AxisFactorType>> meshAxisFactorsMajorFirst;
};

struct StablehloSendToDistributedCollectivePattern
    : public OpRewritePattern<stablehlo::SendOp> {
  using OpRewritePattern<stablehlo::SendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SendOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    // TODO: Lower StableHLO send op to distributed dialect point-to-point op.
    return rewriter.notifyMatchFailure(
        op, "TODO: stablehlo.send to distributed point-to-point rewrite not "
            "implemented yet");
  }
};

struct StablehloRecvToDistributedCollectivePattern
    : public OpRewritePattern<stablehlo::RecvOp> {
  using OpRewritePattern<stablehlo::RecvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::RecvOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    // TODO: Lower StableHLO recv op to distributed dialect point-to-point op.
    return rewriter.notifyMatchFailure(
        op, "TODO: stablehlo.recv to distributed point-to-point rewrite not "
            "implemented yet");
  }
};

struct ShardyToDistributedPass
    : public impl::ShardyToDistributedPassBase<ShardyToDistributedPass> {
  using ShardyToDistributedPassBase::ShardyToDistributedPassBase;

  // For now we simplify our life by requiring all shardy functions to have the
  // same mesh. This will be relaxed at need as implementation progresses.
  sdy::MeshAttr getCommonMesh(const FindShardyFunctionsAnalysis &analysis) {
    auto shardyFunctions = analysis.getShardyFunctions();
    sdy::MeshAttr commonMesh = nullptr;
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         shardyFunctions) {
      if (info.meshes.size() != 1) {
        getOperation()->emitError()
            << "expected shardy function to have exactly one mesh, found "
            << info.meshes.size();
        signalPassFailure();
        return nullptr;
      }
      if (!commonMesh) {
        commonMesh = info.meshes[0];
      } else if (commonMesh != info.meshes[0]) {
        getOperation()->emitError()
            << "expected all shardy functions to have the same mesh, found "
            << commonMesh << " and " << info.meshes[0];
        signalPassFailure();
        return nullptr;
      }
    }
    return commonMesh;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    FailureOr<distributed::PhysicalMeshOp> physicalMesh =
        distributed::findUniquePhysicalMesh(moduleOp);
    if (failed(physicalMesh)) {
      signalPassFailure();
      return;
    }

    const FindShardyFunctionsAnalysis &analysis =
        getAnalysis<FindShardyFunctionsAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }
    if (analysis.getShardyFunctions().empty()) {
      moduleOp.emitRemark() << "no shardy functions found, skipping pass";
      return;
    }

    sdy::MeshAttr commonMesh = getCommonMesh(analysis);
    if (!commonMesh) {
      return;
    }

    // Create a new mesh computation using the modules pysical mesh
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointAfter(
        *physicalMesh); // graph region, doesn't matter
    distributed::MeshComputationOp meshComputation =
        builder.create<distributed::MeshComputationOp>(
            moduleOp.getLoc(), builder.getStringAttr("mesh_computation"),
            FlatSymbolRefAttr::get(physicalMesh->getSymNameAttr()));
    Region &meshComputationBody = meshComputation.getBody();
    if (meshComputationBody.empty()) {
      meshComputationBody.emplaceBlock();
    }
    builder.setInsertionPointToStart(&meshComputationBody.front());

    // Create a new logical mesh inside of the mesh computation using the common
    // shardy mesh for axis sizes. Internally record a mapping from shardy axis
    // names to distributed axis values.
    llvm::SmallVector<int32_t> logicalAxisExtents;
    llvm::SmallVector<std::string> shardyAxisNames;
    for (sdy::MeshAxisAttr axis : commonMesh.getAxes()) {
      int64_t axisSize = axis.getSize();
      if (axisSize <= 0 || axisSize > std::numeric_limits<int32_t>::max()) {
        moduleOp.emitError() << "unsupported shardy mesh axis size " << axisSize
                             << " for axis " << axis.getName();
        signalPassFailure();
        return;
      }
      logicalAxisExtents.push_back(static_cast<int32_t>(axisSize));
      shardyAxisNames.push_back(axis.getName().str());
    }

    distributed::LogicalMeshAxesOp logicalMeshAxes =
        builder.create<distributed::LogicalMeshAxesOp>(
            moduleOp.getLoc(),
            builder.getDenseI32ArrayAttr(logicalAxisExtents));

    llvm::StringMap<Value> shardyToDistributedAxis;
    // Map the canonical logical mesh axis to factor
    // types. Shardy works on full axes so we produce only one factor
    // per axis, which we can subdivide in transformations later if needed.
    // Mapping here determined by arg number.
    auto factors = axis::viewAxesAsFactors(logicalMeshAxes.getAxes(), builder,
                                           moduleOp.getLoc());
    for (auto [idx, factor] : llvm::enumerate(factors)) {
      shardyToDistributedAxis[shardyAxisNames[idx]] = factor;
    }

    auto executionContext =
        builder
            .create<axis::AxisProductOp>(moduleOp.getLoc(), asValues(factors))
            .getProduct();
    auto replicateOver =
        builder.create<axis::AxisProductOp>(moduleOp.getLoc(), ValueRange())
            .getProduct();

    llvm::DenseSet<StringAttr> distributedFunctionNames;
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         analysis.getShardyFunctions()) {
      distributedFunctionNames.insert(info.symName);
    }

    // Now we move the shardy functions into the mesh computation,
    // rewrite them to use the distributed dialect ops for functions,
    // yield, and call, and raise their collectives to our collective
    // abstractions. Normal HLO ops can be left as-is, since they are
    // supposed to be local sizes anyway.
    builder.setInsertionPointToEnd(&meshComputationBody.front());
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         analysis.getShardyFunctions()) {
      func::FuncOp shardyFunction = info.funcOp;
      shardyFunction->moveBefore(&meshComputationBody.front(),
                                 meshComputationBody.front().end());
    }

    RewritePatternSet patterns(moduleOp.getContext());
    patterns.add<FuncToDistributedFunctionPattern>(moduleOp.getContext(),
                                                   executionContext);
    patterns.add<FuncCallToDistributedCallPattern>(
        moduleOp.getContext(), distributedFunctionNames, replicateOver);
    patterns.add<FuncReturnToDistributedYieldPattern>(moduleOp.getContext());
    // Temporarily disabled until getDimMappings specializations are
    // implemented.
    // patterns.add<StablehloCollectiveToDistributedCollectivePattern<
    //                  stablehlo::AllReduceOp>,
    //              StablehloCollectiveToDistributedCollectivePattern<
    //                  stablehlo::AllGatherOp>,
    //              StablehloCollectiveToDistributedCollectivePattern<
    //                  stablehlo::AllToAllOp>,
    //              StablehloCollectiveToDistributedCollectivePattern<
    //                  stablehlo::CollectivePermuteOp>,
    //              StablehloCollectiveToDistributedCollectivePattern<
    //                  stablehlo::ReduceScatterOp>,
    //              StablehloCollectiveToDistributedCollectivePattern<
    //                  stablehlo::CollectiveBroadcastOp>>(
    //     moduleOp.getContext(), shardyToDistributedAxis);
    // patterns.add<StablehloSendToDistributedCollectivePattern,
    //              StablehloRecvToDistributedCollectivePattern>(
    //     moduleOp.getContext());
    patterns.add<StablehloCollectiveToDistributedCollectivePattern<
                     stablehlo::AllReduceOp>,
                 StablehloCollectiveToDistributedCollectivePattern<
                     stablehlo::AllGatherOp>,
                 StablehloCollectiveToDistributedCollectivePattern<
                     stablehlo::CollectivePermuteOp>>(
        moduleOp.getContext(), shardyToDistributedAxis, factors);

    if (failed(applyPatternsGreedily(meshComputation, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    (void)shardyToDistributedAxis;
    (void)logicalMeshAxes;
    (void)factors;
  }
};

} // namespace

} // namespace mlir::enzyme::distributed