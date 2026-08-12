#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

#include <atomic>
#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DISTRIBUTEDTOHLOPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Spatial axes are mesh axes that correspond to distributed placement.
bool isAxisSpatial(Type axisType) {
  return isa<distributed::LogicalMeshAxisType,
             distributed::PhysicalCommAxisType>(axisType);
}

// Collectives with no reduction groups are pure data-movement operations.
bool hasAnyReduction(distributed::DistributedCollectiveOp op) {
  if (op.getReductionGroups().empty()) {
    return false;
  }
  return true;
}

// Restrict all-reduce lowering to exactly one spatial reduction group.
bool hasSingleSpatialReduction(distributed::DistributedCollectiveOp op) {
  // Condition 1: op has exactly 1 reduction group
  auto reductionGroups = axis::castTypedValueList<axis::FactorGroupType>(
      op.getReductionGroups(), "FactorGroupType");
  if (reductionGroups.size() != 1) {
    return false;
  }

  auto reducedFactors = axis::getProductProvenanceFactors(reductionGroups[0]);
  if (failed(reducedFactors)) {
    return false;
  }

  // For each axis of provenance in the reduction group, check that it is a
  // spatial axis.
  for (TypedValue<axis::AxisFactorType> factor : *reducedFactors) {
    axis::AxisFactorType factorType = factor.getType();
    if (!isAxisSpatial(factorType.getAxisType())) {
      return false;
    }
  }
  return true;
}

// StableHLO collective lowering requires the input/output meshes to span the
// same index space.
static bool
hasSameInputOutputMeshIndexSpace(distributed::DistributedCollectiveOp op) {
  auto inputMesh = cast<TypedValue<axis::FactorGroupType>>(op.getInputMesh());
  auto outputMesh = cast<TypedValue<axis::FactorGroupType>>(op.getOutputMesh());

  auto inputMeshFactors = axis::getProductProvenanceFactors(inputMesh);
  auto outputMeshFactors = axis::getProductProvenanceFactors(outputMesh);
  if (failed(inputMeshFactors) || failed(outputMeshFactors)) {
    return false;
  }
  return axis::areFactorIndexSpacesEqual(*inputMeshFactors, *outputMeshFactors);
}

static FailureOr<Attribute> createReplicaGroupsFromPhysicalFactors(
    int64_t flatMeshExtent,
    ArrayRef<TypedValue<axis::AxisFactorType>> reductionFactors,
    OpBuilder &builder) {
  assert(reductionFactors.size() > 0 &&
         "expected at least one reduction factor for replica group creation");

  // Create an axis name to represent the flat index space
  constexpr StringLiteral kFlatAxisName = "axis_0";
  SmallVector<stablehlo::MeshAxisAttr> meshAxes;
  meshAxes.push_back(stablehlo::MeshAxisAttr::get(
      builder.getContext(), kFlatAxisName, flatMeshExtent));

  // For each reduction axis factor, push the corresponding subaxis of the
  // flat index space into the replica group.
  SmallVector<Attribute> reductionAxisRefs;
  reductionAxisRefs.reserve(reductionFactors.size());
  for (TypedValue<axis::AxisFactorType> reducedFactor : reductionFactors) {
    auto reducedAxisType = reducedFactor.getType().getAxisType();
    // Reduced factors must also be over physical mesh axes.
    auto physicalAxisType =
        dyn_cast<distributed::PhysicalCommAxisType>(reducedAxisType);
    if (!physicalAxisType) {
      return failure(
          "Reduction factor not physical axis, cannot project to device ids");
    }

    int64_t axisIdStride = static_cast<int64_t>(physicalAxisType.getIdStride());
    int64_t subExtent = axis::getFactorExtent(reducedFactor);
    int64_t axisLocalStride = axis::getFactorStride(reducedFactor);
    int64_t subStride = axisIdStride * axisLocalStride;

    stablehlo::SubAxisInfoAttr subAxis;
    subAxis = stablehlo::SubAxisInfoAttr::get(builder.getContext(), subStride,
                                              subExtent);
    reductionAxisRefs.push_back(stablehlo::AxisRefAttr::get(
        builder.getContext(), kFlatAxisName, subAxis));
  }

  auto meshAttr = stablehlo::MeshAttr::get(builder.getContext(), meshAxes,
                                           DenseIntElementsAttr());
  return stablehlo::ReplicaGroupMeshAxesAttr::get(
      builder.getContext(), meshAttr, builder.getArrayAttr(reductionAxisRefs));
}

static int64_t getNextStablehloChannelId() {
  // StableHLO channel ids are global within this process and independent of
  // which communication op kind requests the next id.
  static std::atomic<int64_t> nextChannelId{1};
  return nextChannelId.fetch_add(1, std::memory_order_relaxed);
}

static LogicalResult validateScalarReductionFunctionSignature(
    Operation *anchor, FlatSymbolRefAttr reductionFunction,
    Type expectedElementType, RankedTensorType &outScalarTensorType,
    std::string &outError) {
  auto moduleOp = anchor->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    outError = "failed to locate parent module for reduction function lookup";
    return failure();
  }

  auto reductionFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      anchor, reductionFunction);
  if (!reductionFunc) {
    outError = "failed to resolve reduction function symbol";
    return failure();
  }

  FunctionType functionType = reductionFunc.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 1) {
    outError =
        "reduction function must have signature (tensor<elem>, tensor<elem>) "
        "-> tensor<elem>";
    return failure();
  }

  auto lhsType = dyn_cast<RankedTensorType>(functionType.getInput(0));
  auto rhsType = dyn_cast<RankedTensorType>(functionType.getInput(1));
  auto resultType = dyn_cast<RankedTensorType>(functionType.getResult(0));
  if (!lhsType || !rhsType || !resultType) {
    outError = "reduction function operands/result must be ranked tensors";
    return failure();
  }

  if (lhsType.getRank() != 0 || rhsType.getRank() != 0 ||
      resultType.getRank() != 0) {
    outError = "reduction function operands/result must be scalar tensors";
    return failure();
  }

  if (lhsType != rhsType || lhsType != resultType) {
    outError =
        "reduction function scalar tensor operands/result must have identical "
        "types";
    return failure();
  }

  if (lhsType.getElementType() != expectedElementType) {
    outError =
        "reduction function scalar element type must match collective tensor "
        "element type";
    return failure();
  }

  outScalarTensorType = lhsType;
  return success();
}

static stablehlo::ChannelHandleAttr
createUniqueStablehloChannelHandle(Operation *op) {
  // StableHLO collectives use channel type 0.
  constexpr int64_t kCollectiveChannelType = 0;
  int64_t channelId = getNextStablehloChannelId();
  return stablehlo::ChannelHandleAttr::get(op->getContext(), channelId,
                                           kCollectiveChannelType);
}

auto bindBuildReductionComputation(FlatSymbolRefAttr reductionFunc,
                                   RankedTensorType reductionResultType) {
  return [reductionFunc, reductionResultType](RegionBuilder &rb) -> void {
    OpBuilder &builder = rb.getOpBuilder();
    Value lhs = Argument(rb, reductionResultType).getValue();
    Value rhs = Argument(rb, reductionResultType).getValue();
    auto op = builder.create<func::CallOp>(rb.getLoc(), reductionFunc,
                                           TypeRange{reductionResultType},
                                           ValueRange{lhs, rhs});
    builder.create<stablehlo::ReturnOp>(rb.getLoc(), op.getResults());
  };
}

// Replacement AsyncStart builder for a broken generated StableHLO builder
// path: build the op with an explicit future result type before populating the
// body region. This works around the generated StableHLO builder constructing
// the op before the callback can fill the required single-block region.
static MlirOp buildAsyncStart(MlirBuilder &builder, Type resultType,
                              ArrayRef<MlirOp> operands,
                              const RegionBuilderCallback &body) {
  OperationState state(builder.getLoc(),
                       stablehlo::AsyncStartOp::getOperationName());
  state.addOperands(unwrap(operands));
  state.addTypes(resultType);
  state.addRegion();

  auto asyncStartOp =
      cast<stablehlo::AsyncStartOp>(builder.getOpBuilder().create(state));
  RegionBuilder bodyBuilder(builder, asyncStartOp->getRegion(0));
  body(bodyBuilder);
  return MlirOp(builder, asyncStartOp.getResult());
}

// So we can tweak debugging accross patterns easier
auto bindFailWithRemark(distributed::DistributedCollectiveOp op,
                        PatternRewriter &rewriter, StringRef collectiveKind) {
  // mutable because emitRemark() is not const
  return [op, &rewriter,
          collectiveKind](StringRef msg) mutable -> LogicalResult {
    op.emitRemark() << "[distributed-to-hlo match failure: " << collectiveKind
                    << "] " << msg;
    return rewriter.notifyMatchFailure(op, msg);
  };
}

// Validated and prepared async collective inputs.
struct CollectiveInputs {
  Value tensorOperand;
  TensorType inputTensorType;
  TensorType outputTensorType;
};

// Validates and prepares common inputs for async collectives. Validates mesh
// index space alignment and extracts/validates tensor operands and output type.
static LogicalResult validateAndPrepareCollectiveInputs(
    distributed::DistributedCollectiveOp op,
    const std::function<LogicalResult(StringRef)> &failWithRemark,
    CollectiveInputs &result) {
  // Validate that input/output meshes span the same index space
  if (!hasSameInputOutputMeshIndexSpace(op)) {
    return failWithRemark("input/output meshes must have equal index space");
  }

  Value tensorOperand = op.getInputObject();
  auto inputTensorType = dyn_cast<TensorType>(tensorOperand.getType());
  if (!inputTensorType) {
    return failWithRemark("input operand must be a tensor type");
  }

  auto outputTensorType = dyn_cast<TensorType>(op.getOutputType());
  if (!outputTensorType) {
    return failWithRemark("output type must be a tensor type");
  }

  result.tensorOperand = tensorOperand;
  result.inputTensorType = inputTensorType;
  result.outputTensorType = outputTensorType;
  return success();
}

// Validates reduction-specific collective inputs: confirms that reductions
// exist and there is exactly one spatial reduction group.
static LogicalResult validateReductionCollectiveInputs(
    distributed::DistributedCollectiveOp op,
    const std::function<LogicalResult(StringRef)> &failWithRemark) {
  if (!hasAnyReduction(op)) {
    return failWithRemark("collective has no reductions");
  }

  if (!hasSingleSpatialReduction(op)) {
    return failWithRemark("expected exactly one spatial reduction group");
  }

  return success();
}

// Builds an asynchronous collective operation wrapped in StableHLO AsyncStart,
// then replaces the distributed op with an UnrealizedConversionCastOp that
// adapts the future to the distributed async handle type. The collective
// operation is built by the provided callback, which receives the async tensor
// operand to build the appropriate collective op.
static LogicalResult buildAndReplaceAsyncCollective(
    distributed::DistributedCollectiveOp op, const CollectiveInputs &inputs,
    const std::function<void(Value, RegionBuilder &)> &buildCollectiveOp,
    PatternRewriter &rewriter) {
  auto buildAsyncStartRegion = [&](RegionBuilder &rb) -> void {
    Value asyncTensorOperand =
        Argument(rb, inputs.tensorOperand.getType()).getValue();
    buildCollectiveOp(asyncTensorOperand, rb);
  };

  MlirBuilder asyncStartBuilder(rewriter, op.getLoc());
  SmallVector<MlirOp> asyncStartOperands =
      wrap(asyncStartBuilder, ValueRange{inputs.tensorOperand});
  Type asyncStartResultType = stablehlo::FutureType::get(
      op.getContext(), SmallVector<Type>{inputs.outputTensorType});
  MlirOp asyncStart =
      buildAsyncStart(asyncStartBuilder, asyncStartResultType,
                      asyncStartOperands, buildAsyncStartRegion);

  auto convertedHandle = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), TypeRange{op.getAsyncHandle().getType()},
      ValueRange{asyncStart.getValue()});
  rewriter.replaceOp(op, convertedHandle.getResults());
  return success();
}

// Case 1: all-reduce
struct DistributedCollectiveAllReduceToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    auto failWithRemark = bindFailWithRemark(op, rewriter, "all-reduce");

    CollectiveInputs inputs;
    if (failed(
            validateAndPrepareCollectiveInputs(op, failWithRemark, inputs))) {
      return failure();
    }

    if (failed(validateReductionCollectiveInputs(op, failWithRemark))) {
      return failure();
    }

    // Happens when all non-reduction axes have an identity map, all
    // reduction axes are spatial axis, and all reduction axis have a
    // replicate LHS.
    // Relying on the op being in canonical indivisible factor form,
    // we expect to see two types of factors in the map: identity mapping,
    // and spatial --> replicate mapping for each spatial axis.
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> reduction_factors;
    TypedValue<axis::AxisMapType> map_val = op.getMapping();
    // know we should see a non-null value here from the verifier conditions
    axis::AxisMapOp map = map_val.getDefiningOp<axis::AxisMapOp>();
    auto paired_groups = map.getTypedMappingPairs();
    llvm::erase_if(paired_groups, axis::predGroupPairIsIdentity());

    // We expect to see only replicate --> spatial mappings covering
    // the reduction space for an all-reduce.
    for (auto [lhs_group, rhs_group] : paired_groups) {
      auto lhs_factors = axis::getProductProvenanceFactors(lhs_group);
      auto rhs_factors = axis::getProductProvenanceFactors(rhs_group);
      if (failed(lhs_factors) || failed(rhs_factors)) {
        return failWithRemark("failed to extract provenance factors from map");
      }
      // If we see a non-identity mapping, it must be a spatial -->
      // replicate mapping for a reduction axis.
      // Expect single factor in each list
      if (lhs_factors->size() != 1 || rhs_factors->size() != 1) {
        return failWithRemark(
            "non-identity axis.map pair must contain exactly one factor");
      }

      bool rhsSpatialLhsReplication =
          isAxisSpatial((*rhs_factors)[0].getType().getAxisType()) &&
          isa<distributed::ReplicationAxisType>(
              (*lhs_factors)[0].getType().getAxisType());
      if (!rhsSpatialLhsReplication) {
        return failWithRemark(
            "non-identity axis.map pair is not spatial<->replication");
      }
      reduction_factors.push_back((*rhs_factors)[0]);
    }

    // if the reduction axes all mapped to replicate, then we have a valid
    // all-reduce
    auto reduction_groups = axis::castTypedValueList<axis::FactorGroupType>(
        op.getReductionGroups(), "FactorGroupType");
    auto reduced_factors =
        axis::getProductProvenanceFactors(reduction_groups[0]);
    if (failed(reduced_factors)) {
      return failWithRemark("failed to extract reduction provenance factors");
    }

    bool match = axis::areFactorListsStructurallyEqual(reduction_factors,
                                                       *reduced_factors);
    if (!match) {
      return failWithRemark("reduction factors derived from axis.map do not "
                            "match reduces() factors");
    }

    // Lower to an asynchronous stablehlo.all_reduce
    auto inputMesh = cast<TypedValue<axis::FactorGroupType>>(op.getInputMesh());
    auto inputMeshFactors = axis::getProductProvenanceFactors(inputMesh);
    if (failed(inputMeshFactors)) {
      return failWithRemark("failed to extract input mesh provenance factors");
    }
    int64_t flatMeshExtent = 1;
    for (TypedValue<axis::AxisFactorType> meshFactor : *inputMeshFactors) {
      // Input mesh must be a product of physical axes for StableHLO lowering.
      auto physicalAxisType = dyn_cast<distributed::PhysicalCommAxisType>(
          meshFactor.getType().getAxisType());
      if (!physicalAxisType) {
        return failWithRemark(
            "input mesh factor is not a physical communication axis");
      }
      (void)physicalAxisType;
      int64_t axisExtent = axis::getFactorExtent(meshFactor);
      flatMeshExtent *= axisExtent;
    }

    auto replicaGroups = createReplicaGroupsFromPhysicalFactors(
        flatMeshExtent, reduction_factors, rewriter);
    if (failed(replicaGroups)) {
      return failWithRemark("failed to build stablehlo replica groups from "
                            "reduction factors (expected physical axes)");
    }

    auto reductionFunctionsAttr = op.getReductionFunctionsAttr();
    if (!reductionFunctionsAttr || reductionFunctionsAttr.size() != 1) {
      return failWithRemark("expected exactly one reduction function symbol");
    }
    auto reductionFunction =
        dyn_cast<FlatSymbolRefAttr>(reductionFunctionsAttr[0]);
    if (!reductionFunction) {
      return failWithRemark(
          "reduction function must be a flat symbol reference");
    }

    RankedTensorType scalarReductionTensorType;
    std::string signatureError;
    if (failed(validateScalarReductionFunctionSignature(
            op, reductionFunction, inputs.inputTensorType.getElementType(),
            scalarReductionTensorType, signatureError))) {
      return failWithRemark(signatureError);
    }

    auto buildComputation = bindBuildReductionComputation(
        reductionFunction, scalarReductionTensorType);

    auto buildCollectiveOp = [&](Value asyncTensorOperand,
                                 RegionBuilder &rb) -> void {
      SmallVector<MlirOp> reductionOperands =
          wrap(rb, ValueRange{asyncTensorOperand});
      stablehlo::ChannelHandleAttr channelHandle =
          createUniqueStablehloChannelHandle(op);
      SmallVector<MlirOp> hloReduce = stablehlo::AllReduce(
          rb, reductionOperands, buildComputation, *replicaGroups,
          channelHandle, /*use_global_device_ids=*/true);
      assert(hloReduce.size() == 1 &&
             "expected exactly one result from stablehlo.all_reduce");
      rb.getOpBuilder().create<stablehlo::ReturnOp>(rb.getLoc(),
                                                    hloReduce[0].getValue());
    };

    return buildAndReplaceAsyncCollective(op, inputs, buildCollectiveOp,
                                          rewriter);
  }
};

// Case 2: reduce-scatter
struct DistributedCollectiveReduceScatterToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    auto failWithRemark = bindFailWithRemark(op, rewriter, "Reduce-Scatter");

    CollectiveInputs inputs;
    if (failed(
            validateAndPrepareCollectiveInputs(op, failWithRemark, inputs))) {
      return failure();
    }

    if (failed(validateReductionCollectiveInputs(op, failWithRemark))) {
      return failure();
    }
    // Happens when all reduction axes are spatial axes,
    // exactly 1 tensor axis maps to the product of the spatial axes,
    // and all other tensor axes have an identy map.
    // Note: identity map here must account for the differing tensor
    // types on the LHS and RHS. Essentially, aside from the split
    // dimension, we are looking for same rank, extent, and stride.

    auto map_val = op.getMapping();
    axis::AxisMapOp map_op = map_val.getDefiningOp<axis::AxisMapOp>();
    if (!map_op) {
      return failWithRemark(
          "expected axis.map to be defined by an axis.map op");
    }

    auto paired_groups = map_op.getTypedMappingPairs();
    llvm::erase_if(paired_groups, axis::predGroupPairIsIdentity(false));

    // If match, we expect to see one upper range of the tensor mapped to
    // the same space as the reduction axes.
    llvm::SmallVector<std::pair<int, TypedValue<axis::AxisFactorType>>>
        rhs_space_factors;
    int min_tensor_stride = INT_MAX;
    std::optional<TypedValue<axis::ShapeAxisType>> tensor_axis;
    llvm::SmallVector<TypedValue<axis::AxisFactorType>> tensor_factors;
    for (auto [lhs_group, rhs_group] : paired_groups) {
      auto lhs_factors = axis::getProductProvenanceFactors(lhs_group);
      auto rhs_factors = axis::getProductProvenanceFactors(rhs_group);
      if (failed(lhs_factors) || failed(rhs_factors)) {
        return failWithRemark("failed to extract provenance factors from map");
      }
      // Expect single factor in each list due to indivisible factor form
      if (lhs_factors->size() != 1 || rhs_factors->size() != 1) {
        return failWithRemark(
            "non-identity axis.map pair must contain exactly one factor");
      }

      // expect to see tensor --> spatial, on a single tensor axis
      auto this_tensor_axis_factor = (*lhs_factors)[0];
      auto this_tensor_axis_type = dyn_cast<axis::ShapeAxisType>(
          this_tensor_axis_factor.getType().getAxisType());
      if (!this_tensor_axis_type) {
        return failWithRemark(
            "non-identity axis.map pair LHS is not a tensor shape axis");
      }
      // Get the provenance axis (the actual axis value being split)
      auto this_tensor_axis_result =
          axis::getFactorProvenanceAxis(this_tensor_axis_factor);
      if (failed(this_tensor_axis_result)) {
        return failWithRemark("failed to get provenance axis from factor");
      }
      auto this_tensor_axis_typed = *this_tensor_axis_result;

      // Cast the provenance axis value to the expected ShapeAxisType
      auto this_tensor_axis_shape_typed =
          axis::castTypedValue<axis::ShapeAxisType>(
              static_cast<Value>(this_tensor_axis_typed), "ShapeAxisType");

      if (!tensor_axis.has_value()) {
        // Write down the first tensor axis value we see
        tensor_axis = this_tensor_axis_shape_typed;
      } else {
        // Check that all LHS tensor axes are the same
        Value stored_axis_value = *tensor_axis;
        if (stored_axis_value != static_cast<Value>(this_tensor_axis_typed)) {
          return failWithRemark("splitting over multiple tensor axes");
        }
      }
      int this_tensor_stride = axis::getFactorStride((*lhs_factors)[0]);
      min_tensor_stride = std::min(min_tensor_stride, this_tensor_stride);
      tensor_factors.push_back((*lhs_factors)[0]);
      rhs_space_factors.push_back({this_tensor_stride, (*rhs_factors)[0]});
    }

    // Check that we have a contiguous upper range of the tensor axis we are
    // splitting. this is equivalent to covering the whole space upwards of
    // min_tensor_stride
    llvm::SmallVector<std::pair<int, int>> max_factor_pairs =
        axis::build_max_factors(tensor_factors);
    if (max_factor_pairs.size() != 1) {
      return failWithRemark(
          "tensor factors do not cover a contiguous upper range of the "
          "tensor axis");
    }
    int total_extent = max_factor_pairs[0].first * max_factor_pairs[0].second;
    if (!tensor_axis.has_value()) {
      return failWithRemark("no tensor axis was identified for reduce-scatter");
    }
    auto tensor_axis_typed = *tensor_axis;
    int total_expected_extent = axis::getAxisExtent(tensor_axis_typed);
    if (total_extent != total_expected_extent) {
      return failWithRemark("tensor factors does not cover full upper range");
    }

    // Ok, we are sharding over the upper range of a single tensor axis.
    // Now we sort the spatial factors by stride (major/highest first)
    // to get our replica group ordering
    std::sort(
        rhs_space_factors.begin(), rhs_space_factors.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    auto sorted_rhs_space_factors = llvm::to_vector<4>(llvm::map_range(
        rhs_space_factors, [](const auto &pair) { return pair.second; }));

    // Build the reduce_scatter
    FailureOr<Attribute> replicaGroups = createReplicaGroupsFromPhysicalFactors(
        total_expected_extent, sorted_rhs_space_factors, rewriter);
    if (failed(replicaGroups)) {
      return failWithRemark("failed to build stablehlo replica groups from "
                            "reduction factors");
    }
    int scatter_dim = axis::getAxisDimIndex(tensor_axis_typed);

    RankedTensorType scalarReductionTensorType;
    std::string signatureError;
    FlatSymbolRefAttr reductionFunction =
        cast<FlatSymbolRefAttr>(op.getReductionFunctionsAttr()[0]);
    if (failed(validateScalarReductionFunctionSignature(
            op, reductionFunction, inputs.inputTensorType.getElementType(),
            scalarReductionTensorType, signatureError))) {
      return failWithRemark(signatureError);
    }

    auto buildComputation = bindBuildReductionComputation(
        reductionFunction, scalarReductionTensorType);

    auto buildCollectiveOp = [&](Value asyncTensorOperand,
                                 RegionBuilder &rb) -> void {
      SmallVector<MlirOp> reductionOperands =
          wrap(rb, ValueRange{asyncTensorOperand});
      stablehlo::ChannelHandleAttr channelHandle =
          createUniqueStablehloChannelHandle(op);
      MlirBuilder scatterBuilder(rb.getOpBuilder(), rb.getLoc());
      MlirOp scatterOperand = reductionOperands[0];
      MlirOp hloReduce = stablehlo::ReduceScatter(
          inputs.outputTensorType, scatterOperand, buildComputation,
          scatter_dim, *replicaGroups, channelHandle,
          /*use_global_device_ids=*/true);
      rb.getOpBuilder().create<stablehlo::ReturnOp>(rb.getLoc(),
                                                    hloReduce.getValue());
    };

    return buildAndReplaceAsyncCollective(op, inputs, buildCollectiveOp,
                                          rewriter);
  }
};

// Case 3: collective-permute
struct DistributedCollectivePermuteToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    if (hasAnyReduction(op)) {
      return failure();
    }

    // Happens with no reduction axes and all tensor dimensions are identity
    // mapped.
    (void)op;
    (void)rewriter;
    return failure();
  }
};

// Case 4: all-gather
struct DistributedCollectiveAllGatherToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    if (hasAnyReduction(op)) {
      return failure();
    }

    // No reduction axes. Map contains some replicate --> SpaceDim(i),
    // and those same SpaceDim(i) map to a contiguous in-order subdimension
    // of the output tensor (the contatenation dim). All other dimensions
    // identity map (modulo tensor type).
    (void)op;
    (void)rewriter;
    return failure();
  }
};

// Case 5: all-to-all
struct DistributedCollectiveAllToAllToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    if (hasAnyReduction(op)) {
      return failure();
    }

    // No reduction axes, one subaxis moving from space to tensor, one
    // subaxis moving from tensor to space.
    (void)op;
    (void)rewriter;
    return failure();
  }
};

// Case 6: collective-broadcast
struct DistributedCollectiveBroadcastToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    // Not handled here, belongs more with point-to-point communication.
    (void)op;
    (void)rewriter;
    return failure();
  }
};

struct DistributedToStablehloTypeConverter : public TypeConverter {
  // Convert distributed async handles into stablehlo future types.
  DistributedToStablehloTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](distributed::AsynchHandleType type) {
      return stablehlo::FutureType::get(type.getContext(),
                                        SmallVector<Type>{type.getValueType()});
    });

    auto materializeCast = [](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return Value();
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

struct DistributedAwaitToStablehloAsyncDoneConversionPattern
    : public OpConversionPattern<distributed::DistributedAwait> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(distributed::DistributedAwait op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MlirBuilder asyncDoneBuilder(rewriter, op.getLoc());
    MlirOp future(asyncDoneBuilder, adaptor.getAsyncHandle());
    MlirOp asyncDone = stablehlo::AsyncDone(future);
    rewriter.replaceOp(op, asyncDone.getValue());
    return success();
  }
};

static void
populateDistributedCollectiveToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributedCollectiveAllReduceToStablehloPattern,
               DistributedCollectiveReduceScatterToStablehloPattern,
               DistributedCollectivePermuteToStablehloPattern,
               DistributedCollectiveAllGatherToStablehloPattern,
               DistributedCollectiveAllToAllToStablehloPattern,
               DistributedCollectiveBroadcastToStablehloPattern>(
      patterns.getContext());
}

static void populateDistributedAwaitToStablehloConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<DistributedAwaitToStablehloAsyncDoneConversionPattern>(
      typeConverter, patterns.getContext());
}

struct DistributedToHloPass
    : public impl::DistributedToHloPassBase<DistributedToHloPass> {
  using DistributedToHloPassBase::DistributedToHloPassBase;

  void runOnOperation() override {
    // First lower collectives greedily, then convert await via type
    // conversion.
    RewritePatternSet collectivePatterns(&getContext());
    populateDistributedCollectiveToStablehloPatterns(collectivePatterns);

    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(collectivePatterns)))) {
      signalPassFailure();
      return;
    }

    DistributedToStablehloTypeConverter typeConverter;
    RewritePatternSet conversionPatterns(&getContext());
    populateDistributedAwaitToStablehloConversionPatterns(typeConverter,
                                                          conversionPatterns);

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<distributed::DistributedAwait>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(conversionPatterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
