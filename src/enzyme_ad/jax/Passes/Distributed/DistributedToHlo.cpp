#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <atomic>

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

static FailureOr<Attribute> createReplicaGroupsFromPhysicalReductionFactors(
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

static stablehlo::ChannelHandleAttr
createUniqueStablehloChannelHandle(Operation *op) {
  // StableHLO collectives use channel type 0.
  constexpr int64_t kCollectiveChannelType = 0;
  int64_t channelId = getNextStablehloChannelId();
  return stablehlo::ChannelHandleAttr::get(op->getContext(), channelId,
                                           kCollectiveChannelType);
}

// StableHLO reduction ops expect a 2-arg function to be embedded.
// Our distributed collective have a function symbol reference.
// Create the region to call into the reduction function.
template <typename StablehloReductionOp>
static void
buildStableHLOReductionFunction(StablehloReductionOp reductionOp,
                                FlatSymbolRefAttr reductionFunction,
                                OpBuilder &builder) {
  // Materialize a 2-arg scalar reduction region expected by stablehlo ops.
  auto inputTensorType =
      dyn_cast<ShapedType>(reductionOp->getOperand(0).getType());
  assert(inputTensorType &&
         "stablehlo reduction operand must be shaped for scalarization");

  auto scalarTensorType =
      RankedTensorType::get({}, inputTensorType.getElementType());

  Region &computation = reductionOp.getComputation();
  Block *body = nullptr;
  if (computation.empty()) {
    body = new Block();
    computation.push_back(body);
  } else {
    body = &computation.front();
  }

  if (body->empty()) {
    body->addArgument(scalarTensorType, reductionOp.getLoc());
    body->addArgument(scalarTensorType, reductionOp.getLoc());
  }
  if (!body->empty() && isa<stablehlo::ReturnOp>(body->back())) {
    body->back().erase();
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  if (!SymbolTable::lookupNearestSymbolFrom(reductionOp, reductionFunction)) {
    auto add = builder.create<stablehlo::AddOp>(
        reductionOp.getLoc(), body->getArgument(0), body->getArgument(1));
    builder.create<stablehlo::ReturnOp>(reductionOp.getLoc(), add.getResult());
    return;
  }

  auto call = builder.create<func::CallOp>(
      reductionOp.getLoc(), reductionFunction.getValue(),
      TypeRange{scalarTensorType},
      ValueRange{body->getArgument(0), body->getArgument(1)});
  builder.create<stablehlo::ReturnOp>(reductionOp.getLoc(), call.getResults());
}

static stablehlo::AsyncStartOp createStablehloAsyncStartOp(
    PatternRewriter &rewriter, Location loc, Type resultType,
    ValueRange operands,
    llvm::function_ref<Value(OpBuilder &, Location, ValueRange)> buildBody) {
  // Build async_start with a single-block body and caller-supplied payload op.
  auto futureType = stablehlo::FutureType::get(rewriter.getContext(),
                                               SmallVector<Type>{resultType});
  auto asyncStart =
      rewriter.create<stablehlo::AsyncStartOp>(loc, futureType, operands);

  Block *bodyBlock = &asyncStart.getBody().front();

  if (bodyBlock->empty()) {
    for (Value operand : operands) {
      bodyBlock->addArgument(operand.getType(), loc);
    }
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(bodyBlock);
  Value bodyResult = buildBody(rewriter, loc, bodyBlock->getArguments());
  assert(
      (!bodyBlock->empty() && !isa<stablehlo::ReturnOp>(bodyBlock->back())) &&
      "async_start buildBody must not emit stablehlo.return; helper owns the "
      "return");
  rewriter.create<stablehlo::ReturnOp>(loc, bodyResult);
  return asyncStart;
}

// Case 1: all-reduce
struct DistributedCollectiveAllReduceToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    auto failWithRemark = [&](StringRef msg) -> LogicalResult {
      op.emitRemark() << "[distributed-to-hlo all-reduce match failure] "
                      << msg;
      return rewriter.notifyMatchFailure(op, msg);
    };

    if (!hasSameInputOutputMeshIndexSpace(op)) {
      return failWithRemark("input/output meshes must have equal index space");
    }

    if (!hasAnyReduction(op)) {
      return failWithRemark("collective has no reductions");
    }

    // all reductions currently expect only
    // spatial reductions for lowering
    if (!hasSingleSpatialReduction(op)) {
      return failWithRemark("expected exactly one spatial reduction group");
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
    for (auto [lhs, rhs] :
         llvm::zip(map.getMappingLhs(), map.getMappingRhs())) {
      auto lhs_group = dyn_cast<TypedValue<axis::FactorGroupType>>(lhs);
      auto rhs_group = dyn_cast<TypedValue<axis::FactorGroupType>>(rhs);
      if (!lhs_group || !rhs_group) {
        return failWithRemark(
            "axis.map entries must be factor-group typed values");
      }

      auto lhs_factors = axis::getProductProvenanceFactors(lhs_group);
      auto rhs_factors = axis::getProductProvenanceFactors(rhs_group);
      if (failed(lhs_factors) || failed(rhs_factors)) {
        return failWithRemark("failed to extract axis.map provenance factors");
      }
      if (axis::areFactorListsStructurallyEqual(*lhs_factors, *rhs_factors)) {
        continue;
      }
      // If we see a non-identity mapping, it must be a spatial -->
      // replicate mapping for a reduction axis.
      // Expect single factor in each list
      if (lhs_factors->size() != 1 || rhs_factors->size() != 1) {
        return failWithRemark(
            "non-identity axis.map pair must contain exactly one factor");
      }

      bool lhsSpatialRhsReplication =
          isAxisSpatial((*lhs_factors)[0].getType().getAxisType()) &&
          isa<distributed::ReplicationAxisType>(
              (*rhs_factors)[0].getType().getAxisType());
      bool rhsSpatialLhsReplication =
          isAxisSpatial((*rhs_factors)[0].getType().getAxisType()) &&
          isa<distributed::ReplicationAxisType>(
              (*lhs_factors)[0].getType().getAxisType());
      if (!lhsSpatialRhsReplication && !rhsSpatialLhsReplication) {
        return failWithRemark(
          "non-identity axis.map pair is not spatial<->replication");
      }
      reduction_factors.push_back(lhsSpatialRhsReplication ? (*lhs_factors)[0]
                                                           : (*rhs_factors)[0]);
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
      return failWithRemark(
        "reduction factors derived from axis.map do not match reduces() factors");
    }

    // Now lower into an asynchrounous stablehlo
    // all-reduce op.
    // operands: just the tensor operand of the collective op
    // replica_groups: we need to convert the reduction axes
    // to the replica groups. This is fairly common, so we want
    // a utility that takes a list of factors on PHYSICAL AXES ONLY
    // and converts to a stablehlo ReplicaGroupMeshAxes using the
    // subaxis info ~= stride and extent of the factors AND of their
    // underlying axes for proper ordering / numbering.
    // channel_id: use llvm unique identifier and turn it into a channel handle
    // Reduction takes a region builder for reduction function, we can just
    // splice in a call op to the function symbol we should have on hand.
    // We will need this for reduce_Scatter as well so implement the region
    // builder generically.

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

    auto replicaGroups = createReplicaGroupsFromPhysicalReductionFactors(
        flatMeshExtent, reduction_factors, rewriter);
    if (failed(replicaGroups)) {
      return failWithRemark(
        "failed to build stablehlo replica groups from reduction factors (expected physical axes)");
    }

    auto reductionFunctionsAttr = op.getReductionFunctionsAttr();
    if (!reductionFunctionsAttr || reductionFunctionsAttr.size() != 1) {
      return failWithRemark("expected exactly one reduction function symbol");
    }
    auto reductionFunction =
        dyn_cast<FlatSymbolRefAttr>(reductionFunctionsAttr[0]);
    if (!reductionFunction) {
      return failWithRemark("reduction function must be a flat symbol reference");
    }

    Value tensorOperand = op.getInputObject();
    auto outputTensorType = dyn_cast<TensorType>(op.getOutputTensorType());
    if (!outputTensorType) {
      return failWithRemark("output type must be a tensor type");
    }

    auto asyncStart = createStablehloAsyncStartOp(
        rewriter, op.getLoc(), outputTensorType, ValueRange{tensorOperand},
        [&](OpBuilder &bodyBuilder, Location bodyLoc,
            ValueRange bodyOperands) -> Value {
          auto allReduce = stablehlo::AllReduceOp::create(
              bodyBuilder, bodyLoc, TypeRange{outputTensorType},
              ValueRange{bodyOperands[0]}, *replicaGroups,
              createUniqueStablehloChannelHandle(op),
              /*use_global_device_ids=*/true);
          buildStableHLOReductionFunction(allReduce, reductionFunction,
                                          bodyBuilder);
          return allReduce.getResult(0);
        });

    auto convertedHandle = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), TypeRange{op.getAsyncHandle().getType()},
        ValueRange{asyncStart.getResult()});
    rewriter.replaceOp(op, convertedHandle.getResults());
    return success();
  }
};

// Case 2: reduce-scatter
struct DistributedCollectiveReduceScatterToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasAnyReduction(op)) {
      return failure();
    }

    // all reductions currently expect only
    // spatial reductions for lowering
    if (!hasSingleSpatialReduction(op)) {
      return failure();
    }

    // Happens when all reduction axes are spatial axes,
    // exactly 1 tensor axis maps to the product of the spatial axes,
    // and all other tensor axes have an identy map.
    // Note: identity map here must account for the differing tensor
    // types on the LHS and RHS. Essentially, aside from the split
    // dimension, we are looking for same rank, extent, and stride.
    (void)op;
    (void)rewriter;
    return failure();
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
    Value future = adaptor.getAsyncHandle();

    auto asyncDone = rewriter.create<stablehlo::AsyncDoneOp>(
        op.getLoc(), op.getValue().getType(), future);
    rewriter.replaceOp(op, asyncDone.getResult());
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
    // First lower collectives greedily, then convert await via type conversion.
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
