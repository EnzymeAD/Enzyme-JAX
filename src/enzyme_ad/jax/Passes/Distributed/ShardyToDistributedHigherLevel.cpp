#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include <cctype>
#include <optional>
#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDHIGHERLEVELPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

template <typename RangeT>
static llvm::SmallVector<Value> asValues(const RangeT &groups) {
  llvm::SmallVector<Value> values;
  values.reserve(groups.size());
  for (Value group : groups) {
    values.push_back(group);
  }
  return values;
}

static void
dumpLogicalAxesForMainBlock(Block *mainBlock,
                            ShardyLogicalAxisAnalysis &axisAnalysis) {
  llvm::errs()
      << "[ShardyToDistributedHigherLevel] logical axes for main block\n";
  for (BlockArgument arg : mainBlock->getArguments()) {
    auto partitioningAxes = axisAnalysis.getTensorPartitionDims(arg);
    llvm::errs() << "  arg: " << arg << "\n";
    llvm::errs() << "    partitioning axes: ";
    for (const auto &axis : *partitioningAxes) {
      llvm::errs() << "[";
      for (const auto &symbol : axis) {
        llvm::errs() << "a" << symbol.getId() << ":" << symbol.getExtent()
                     << " ";
      }
      llvm::errs() << "] ";
    }
    llvm::errs() << "\n";
  }
  for (Operation &op : mainBlock->getOperations()) {
    auto partitioningAxes = axisAnalysis.getPartitioningAxes(&op);

    llvm::errs() << "  op: " << op << "\n";
    llvm::errs() << "    partitioning axes: ";
    for (const auto &axis : partitioningAxes) {
      llvm::errs() << "[";
      for (const auto &symbol : axis) {
        llvm::errs() << "a" << symbol.getId() << ":" << symbol.getExtent()
                     << " ";
      }
      llvm::errs() << "] ";
    }
    llvm::errs() << "\n";
  }
}

static std::string getTypeSuffix(Type type) {
  std::string suffix;
  llvm::raw_string_ostream os(suffix);
  type.print(os);
  os.flush();
  for (char &c : suffix) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
  return suffix;
}

mlir::RankedTensorType
toLocalType(mlir::RankedTensorType globalType,
            llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
  auto globalShape = globalType.getShape();
  assert(globalShape.size() == partitioningAxes.size() &&
         "global shape and partitioning axes must have the same rank");
  llvm::SmallVector<int64_t> localShape;
  localShape.reserve(globalShape.size());
  for (size_t i = 0; i < globalShape.size(); ++i) {
    int64_t globalDim = globalShape[i];
    uint64_t extent = 1;
    for (const auto &symbol : partitioningAxes[i]) {
      extent *= symbol.getExtent();
    }
    assert(globalDim % extent == 0 &&
           "global dimension must be divisible by partitioning extent");
    localShape.push_back(globalDim / extent);
  }
  return mlir::RankedTensorType::get(localShape, globalType.getElementType());
}

struct ShardyToDistributedHigherLevelPass
    : public impl::ShardyToDistributedHigherLevelPassBase<
          ShardyToDistributedHigherLevelPass> {
  using ShardyToDistributedHigherLevelPassBase::
      ShardyToDistributedHigherLevelPassBase;
  using TV_AxisFactor = mlir::TypedValue<mlir::enzyme::axis::AxisFactorType>;
  using TV_Axis = mlir::TypedValue<mlir::enzyme::axis::AxisTypeInterface>;
  using TV_FactorGroup = mlir::TypedValue<mlir::enzyme::axis::FactorGroupType>;

  llvm::DenseMap<AxisSymbol, TV_AxisFactor> symbolToLogicalAxis;
  ShardyLogicalAxisAnalysis axisAnalysis;

  struct ShardConflict {
    OpResult value;
    OpShardingRuleAndReductionKind shardingRule;
    llvm::SmallVector<llvm::SmallVector<AxisSymbol>> producerPartitioningAxes;
    RankedTensorType globalType;
    llvm::SmallVector<AxisSymbol> reductionAxes;
    llvm::SmallVector<OpOperand *> conflictingUses;
    llvm::SmallVector<OpOperand *> nonConflictingUses;
    Value currentValue;
    RankedTensorType currentValueType;
  };

  // optional just so we can have a default constructor. We will set this in
  // runOnOperation.
  std::optional<OpBuilder> axis_builder;
  std::optional<mlir::Location> axis_loc;

  ShardyToDistributedHigherLevelPass() = default;

  TV_AxisFactor getOrCreateLogicalAxisForSymbol(AxisSymbol symbol) {
    // if present, return
    auto it = symbolToLogicalAxis.find(symbol);
    if (it != symbolToLogicalAxis.end()) {
      return it->second;
    }
    // otherwise, we need to instantiate a new logical axis,
    // then turn it into a factor.
    auto op =
        axis_builder->create<mlir::enzyme::distributed::LogicalMeshAxesOp>(
            *axis_loc, symbol.getExtent());
    auto as_factor =
        axis::viewAxesAsFactors(op.getAxes(), *axis_builder, *axis_loc);
    symbolToLogicalAxis[symbol] = as_factor.front();
    return as_factor.front();
  }

  llvm::SmallVector<TV_AxisFactor>
  getLogicalAxesForSymbols(llvm::ArrayRef<AxisSymbol> symbols) {
    llvm::SmallVector<TV_AxisFactor> logical_axes;
    logical_axes.reserve(symbols.size());
    for (const auto &symbol : symbols) {
      logical_axes.push_back(getOrCreateLogicalAxisForSymbol(symbol));
    }
    return logical_axes;
  }

  TV_FactorGroup toLocallyTypedAxisProduct(
      mlir::RankedTensorType local_type,
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    assert(local_type.getRank() == partitioningAxes.size() &&
           "local type and partitioning axes must have the same rank");
    llvm::SmallVector<TV_AxisFactor> factors;
    for (auto [axis_index, axis_symbols] : llvm::enumerate(partitioningAxes)) {
      assert(axis_symbols.size() > 0 && "partitioning axes must be non-empty");
      auto logical_axes = getLogicalAxesForSymbols(axis_symbols);
      for (const TV_AxisFactor &factor : logical_axes) {
        factors.push_back(factor);
      }
      // create a tensor axis factor for the local dimension
      auto local_axis = axis_builder->create<mlir::enzyme::axis::AxisGetAxisOp>(
          *axis_loc, local_type, axis_index);
      llvm::SmallVector<TV_Axis> local_axes = {local_axis.getAxis()};
      auto local_axis_factor =
          axis::viewAxesAsFactors(local_axes, *axis_builder, *axis_loc);
      factors.push_back(local_axis_factor.front());
    }
    // create a factor group for the product of all factors
    auto factor_group = axis_builder->create<mlir::enzyme::axis::AxisProductOp>(
        *axis_loc, asValues(factors));
    return factor_group.getProduct();
  }

  TV_FactorGroup getMeshForTensorPartitioning(
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    // The mesh is the product of the logical axis factors used to shard this
    // specific tensor value.
    llvm::SmallVector<llvm::SmallVector<TV_AxisFactor>> logical_axes;
    logical_axes.reserve(partitioningAxes.size());
    for (const auto &axis : partitioningAxes) {
      logical_axes.push_back(getLogicalAxesForSymbols(axis));
    }
    auto product = axis_builder->create<mlir::enzyme::axis::AxisProductOp>(
        *axis_loc, asValues(flattenNested(logical_axes)));
    return product.getProduct();
  }

  // Normalizes one logical symbol into a single-factor group for
  // distributed.function partitioning_axes metadata.
  TV_FactorGroup getOrCreatePartitioningAxisGroup(AxisSymbol symbol) {
    auto factor = getOrCreateLogicalAxisForSymbol(symbol);
    return axis::viewFactorsAsProduct(ValueRange{factor}, *axis_builder,
                                      *axis_loc);
  }

  IndexedTensorShardingAttr buildIndexedShardingAttr(
      RankedTensorType tensorType,
      const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
          &partitioningDims,
      llvm::DenseMap<AxisSymbol, int64_t> &symbolToPartitioningAxisIdx) {
    auto *ctx = &getContext();
    SmallVector<DenseI64ArrayAttr> dimPartitioningAxes;
    dimPartitioningAxes.reserve(tensorType.getRank());

    for (int64_t dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
      SmallVector<int64_t> partitioningAxisIndices;
      if (dimIdx < static_cast<int64_t>(partitioningDims.size())) {
        for (AxisSymbol symbol : partitioningDims[dimIdx]) {
          auto [it, inserted] = symbolToPartitioningAxisIdx.try_emplace(
              symbol, symbolToPartitioningAxisIdx.size());
          (void)inserted;
          partitioningAxisIndices.push_back(it->second);
        }
      }
      dimPartitioningAxes.push_back(
          DenseI64ArrayAttr::get(ctx, partitioningAxisIndices));
    }

    return IndexedTensorShardingAttr::get(
        ctx, dimPartitioningAxes,
        DenseI64ArrayAttr::get(ctx, llvm::ArrayRef<int64_t>{}));
  }

  LogicalResult convertMainToDistributedFunction(ModuleOp moduleOp,
                                                 func::FuncOp mainFunc,
                                                 Block *&mainBlock,
                                                 Operation *&mainScopeOp) {
    auto *ctx = &getContext();
    auto returnOp = dyn_cast<func::ReturnOp>(mainBlock->getTerminator());
    if (!returnOp) {
      mainFunc.emitError() << "expected main terminator to be func.return";
      return failure();
    }

    llvm::DenseMap<AxisSymbol, int64_t> symbolToPartitioningAxisIdx;

    SmallVector<IndexedTensorShardingAttr> argumentShardings;
    argumentShardings.reserve(mainBlock->getNumArguments());
    for (BlockArgument arg : mainBlock->getArguments()) {
      auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType) {
        argumentShardings.push_back(IndexedTensorShardingAttr::get(
            ctx, SmallVector<DenseI64ArrayAttr>{},
            DenseI64ArrayAttr::get(ctx, llvm::ArrayRef<int64_t>{})));
        continue;
      }

      auto maybePartitioning = axisAnalysis.getTensorPartitionDims(arg);
      if (!maybePartitioning) {
        mainFunc.emitError()
            << "missing partitioning mapping for block argument "
            << arg.getArgNumber();
        return failure();
      }

      argumentShardings.push_back(buildIndexedShardingAttr(
          tensorType, *maybePartitioning, symbolToPartitioningAxisIdx));
    }

    SmallVector<IndexedTensorShardingAttr> outputShardings;
    outputShardings.reserve(returnOp.getNumOperands());
    for (OpOperand &operand : returnOp->getOpOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!tensorType) {
        outputShardings.push_back(IndexedTensorShardingAttr::get(
            ctx, SmallVector<DenseI64ArrayAttr>{},
            DenseI64ArrayAttr::get(ctx, llvm::ArrayRef<int64_t>{})));
        continue;
      }

      auto maybePartitioning = axisAnalysis.getTensorPartitionDims(operand);
      if (!maybePartitioning) {
        mainFunc.emitError() << "missing partitioning mapping for return "
                                "operand "
                             << operand.getOperandNumber();
        return failure();
      }

      outputShardings.push_back(buildIndexedShardingAttr(
          tensorType, *maybePartitioning, symbolToPartitioningAxisIdx));
    }

    // Rebuild the partitioning axes in index order.
    SmallVector<Value> orderedPartitioningAxes(
        symbolToPartitioningAxisIdx.size());
    for (const auto &[symbol, idx] : symbolToPartitioningAxisIdx) {
      orderedPartitioningAxes[idx] = getOrCreatePartitioningAxisGroup(symbol);
    }

    auto argShardingsAttr = IndexedTensorShardingPerValueAttr::get(
        ctx, argumentShardings);
    auto outputShardingsAttr =
        IndexedTensorShardingPerValueAttr::get(ctx, outputShardings);

    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPoint(mainFunc);

    auto distributedFunction = builder.create<DistributedFunctionOp>(
        mainFunc.getLoc(), mainFunc.getSymNameAttr(),
        TypeAttr::get(mainFunc.getFunctionType()),
        ValueRange(orderedPartitioningAxes), argShardingsAttr,
        outputShardingsAttr, mainFunc.getSymVisibilityAttr(),
        mainFunc.getArgAttrsAttr(), mainFunc.getResAttrsAttr());
    distributedFunction.getBody().takeBody(mainFunc.getBody());
    mainBlock = &distributedFunction.getBody().front();

    auto movedReturnOp = dyn_cast<func::ReturnOp>(mainBlock->getTerminator());
    if (!movedReturnOp) {
      distributedFunction.emitError()
          << "expected moved main block to end with func.return";
      return failure();
    }

    builder.setInsertionPoint(movedReturnOp);
    auto yieldOp = builder.create<DistributedYieldOp>(movedReturnOp.getLoc(),
                                                      movedReturnOp.getOperands());
    axisAnalysis.markRewrite(movedReturnOp, yieldOp);
    movedReturnOp.erase();

    mainScopeOp = distributedFunction;
    mainFunc.erase();
    return success();
  }

  // Removes existing `sdy.reshard` ops and forwards their input values.
  // We rebuild reshards as explicit collectives later once all conflicts are
  // known.
  void removeExistingReshards(Operation *scopeOp) {
    std::vector<sdy::ReshardOp> toRemove;
    scopeOp->walk([&](sdy::ReshardOp reshardOp) {
      toRemove.push_back(reshardOp);
      reshardOp.getResult().replaceAllUsesWith(reshardOp.getInput());
    });
    for (sdy::ReshardOp reshardOp : toRemove) {
      reshardOp.erase();
    }
  }

  // Scans SSA uses to find values whose producer/consumer sharding disagree
  // and values that carry reduction axes requiring collective insertion.
  std::vector<ShardConflict> collectShardConflicts(Block *mainBlock) {
    std::vector<ShardConflict> conflicts;
    for (Operation &op : mainBlock->getOperations()) {
      for (OpResult result : op.getResults()) {
        auto maybeProducerSharded = axisAnalysis.getTensorPartitionDims(result);
        if (!maybeProducerSharded) {
          op.emitRemark("Found non-sharded result number ")
              << result.getResultNumber() << " of op " << op;
          continue;
        }
        ShardConflict conflict;
        conflict.value = result;
        conflict.shardingRule =
            getOrSynthesizeOpShardingRule(result.getOwner());
        conflict.producerPartitioningAxes = *maybeProducerSharded;
        conflict.globalType = dyn_cast<RankedTensorType>(result.getType());
        conflict.reductionAxes = axisAnalysis.getReductionAxes(result);
        for (OpOperand &use : result.getUses()) {
          auto maybeConsumerSharded = axisAnalysis.getTensorPartitionDims(use);
          if (!maybeConsumerSharded) {
            use.getOwner()->emitRemark(
                "Found non-sharded use of result number ")
                << result.getResultNumber() << " of op " << op;
            continue;
          }
          if (maybeProducerSharded != maybeConsumerSharded) {
            conflict.conflictingUses.push_back(&use);
          } else {
            conflict.nonConflictingUses.push_back(&use);
          }
        }
        if (!conflict.conflictingUses.empty() ||
            !conflict.reductionAxes.empty()) {
          conflicts.push_back(std::move(conflict));
        }
      }
    }
    return conflicts;
  }

  // Replaces one operand use and inserts a cast only when the type must match.
  void rewriteUseWithValue(OpBuilder &builder, Location loc, OpOperand *use,
                           Value replacement) {
    Type expectedUseType = use->get().getType();
    Value valueForUse = replacement;
    if (valueForUse.getType() != expectedUseType) {
      builder.setInsertionPoint(use->getOwner());
      valueForUse = builder
                        .create<UnrealizedConversionCastOp>(
                            loc, expectedUseType, valueForUse)
                        .getResult(0);
    }
    use->set(valueForUse);
  }

  // Builds one collective+await pair and optionally clones a reduction region.
  Value createCollectiveForConflict(
      OpBuilder &builder, ShardConflict &conflict, TV_FactorGroup lhsMesh,
      TV_FactorGroup lhsDims, TV_FactorGroup rhsMesh, TV_FactorGroup rhsDims,
      Type collectiveOutputType, Value collectiveInput,
      llvm::ArrayRef<TV_AxisFactor> collectiveReductionDims) {
    llvm::SmallVector<Value> reductionGroupValues;
    if (!collectiveReductionDims.empty()) {
      auto reductionGroup =
          builder
              .create<mlir::enzyme::axis::AxisProductOp>(
                  conflict.value.getLoc(), asValues(collectiveReductionDims))
              .getProduct();
      reductionGroupValues.push_back(reductionGroup);
    }

    auto mapping = builder.create<mlir::enzyme::axis::AxisMapOp>(
        conflict.value.getLoc(), ValueRange{lhsDims}, ValueRange{rhsDims});
    auto collectiveAndAwait =
        mlir::enzyme::distributed::createCollectiveAndAwait(
            builder, conflict.value.getLoc(), collectiveInput, lhsMesh, rhsMesh,
            ValueRange(reductionGroupValues), mapping.getMap(),
            collectiveOutputType);

    if (!collectiveReductionDims.empty()) {
      if (!conflict.globalType) {
        conflict.value.getOwner()->emitError()
            << "expected ranked tensor type when materializing reduction "
               "collective";
        return Value();
      }
      auto *reductionBody = conflict.shardingRule.getReductionBody(
          conflict.globalType.getElementType());
      if (!reductionBody) {
        conflict.value.getOwner()->emitError()
            << "expected reduction body metadata when materializing reduction "
               "collective";
        return Value();
      }
      IRMapping mapper;
      reductionBody->cloneInto(
          &collectiveAndAwait.collective.getReductionBodies()[0], mapper);
    }

    return collectiveAndAwait.await.getValue();
  }

  // Stage 1: materialize producer-layout reduction collectives and update
  // state.
  LogicalResult
  materializeCollectivesForReductions(ModuleOp moduleOp,
                                      std::vector<ShardConflict> &conflicts) {
    OpBuilder builder(moduleOp.getContext());

    for (ShardConflict &conflict : conflicts) {
      if (!conflict.globalType) {
        conflict.value.getOwner()->emitError()
            << "Found non-ranked tensor type for sharded value "
            << conflict.value;
        return failure();
      }

      auto localType =
          toLocalType(conflict.globalType, conflict.producerPartitioningAxes);
      builder.setInsertionPointAfterValue(conflict.value);
      auto unrealizedCast = builder.create<UnrealizedConversionCastOp>(
          conflict.value.getLoc(), localType, conflict.value);
      conflict.currentValue = unrealizedCast.getResult(0);
      conflict.currentValueType = localType;

      if (conflict.reductionAxes.empty()) {
        continue;
      }

      auto reductionDims = getLogicalAxesForSymbols(conflict.reductionAxes);
      auto lhsDims = toLocallyTypedAxisProduct(
          localType, conflict.producerPartitioningAxes);
      llvm::SmallVector<llvm::SmallVector<AxisSymbol>> lhsPartitioningAxes(
          conflict.producerPartitioningAxes.begin(),
          conflict.producerPartitioningAxes.end());
      lhsPartitioningAxes.push_back(llvm::SmallVector<AxisSymbol>(
          conflict.reductionAxes.begin(), conflict.reductionAxes.end()));
      auto lhsMesh = getMeshForTensorPartitioning(lhsPartitioningAxes);
      auto rhsMesh =
          getMeshForTensorPartitioning(conflict.producerPartitioningAxes);

      conflict.value.getOwner()->emitRemark("Found reduction axes for value ")
          << conflict.value << " of op " << conflict.value.getOwner()
          << ", inserting collective";

      Value reducedValue = createCollectiveForConflict(
          builder, conflict, lhsMesh, lhsDims, rhsMesh, lhsDims, localType,
          conflict.currentValue, reductionDims);
      if (!reducedValue) {
        return failure();
      }

      conflict.currentValue = reducedValue;
      conflict.reductionAxes.clear();
      for (OpOperand *use : conflict.nonConflictingUses) {
        rewriteUseWithValue(builder, conflict.value.getLoc(), use,
                            conflict.currentValue);
      }
      // conflicting will be added in materializeCollectivesForConflicts
      // when they look at the currentValue to find the already reduced ones.
    }

    return success();
  }

  // Stage 2: materialize layout collectives for uses that still conflict.
  LogicalResult
  materializeCollectivesForConflicts(ModuleOp moduleOp,
                                     std::vector<ShardConflict> &conflicts) {
    OpBuilder builder(moduleOp.getContext());

    for (ShardConflict &conflict : conflicts) {
      if (!conflict.currentValue || !conflict.globalType ||
          !conflict.currentValueType) {
        conflict.value.getOwner()->emitError()
            << "missing conflict state before layout collective "
               "materialization";
        return failure();
      }

      auto lhsMesh =
          getMeshForTensorPartitioning(conflict.producerPartitioningAxes);
      auto lhsDims = toLocallyTypedAxisProduct(
          conflict.currentValueType, conflict.producerPartitioningAxes);

      for (OpOperand *use : conflict.nonConflictingUses) {
        rewriteUseWithValue(builder, conflict.value.getLoc(), use,
                            conflict.currentValue);
      }

      for (OpOperand *use : conflict.conflictingUses) {
        builder.setInsertionPoint(use->getOwner());
        auto rhsPartitioningAxes = axisAnalysis.getTensorPartitionDims(*use);
        if (!rhsPartitioningAxes) {
          use->getOwner()->emitError()
              << "missing partitioning axes for conflicting use";
          return failure();
        }

        auto rhsLocalType =
            toLocalType(conflict.globalType, *rhsPartitioningAxes);
        auto rhsDims =
            toLocallyTypedAxisProduct(rhsLocalType, *rhsPartitioningAxes);
        auto rhsMesh = getMeshForTensorPartitioning(*rhsPartitioningAxes);
        Value collective = createCollectiveForConflict(
            builder, conflict, lhsMesh, lhsDims, rhsMesh, rhsDims, rhsLocalType,
            conflict.currentValue, {});
        if (!collective) {
          return failure();
        }
        rewriteUseWithValue(builder, conflict.value.getLoc(), use, collective);
      }
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();

    func::FuncOp mainFunc;
    for (func::FuncOp func : module_op.getOps<func::FuncOp>()) {
      if (func.getName() == "main") {
        mainFunc = func;
        break;
      }
    }

    if (!mainFunc) {
      emitWarning(module_op.getLoc())
          << "no main function found; skipping pass";
      return;
    }

    Region &body = mainFunc.getBody();
    if (body.empty()) {
      emitError(mainFunc.getLoc()) << "main function has no body";
      signalPassFailure();
      return;
    }

    Block *mainBlock = nullptr;
    if (body.getBlocks().size() != 1) {
      emitError(mainFunc.getLoc())
          << "main function must have exactly one block";
      signalPassFailure();
      return;
    }

    mainBlock = &body.front();
    if (!mainBlock) {
      emitError(mainFunc.getLoc()) << "main function has no entry block";
      signalPassFailure();
      return;
    }

    // prep for building axes
    axis_builder = OpBuilder(module_op.getContext());
    axis_builder->setInsertionPointToStart(&module_op.getBodyRegion().front());
    axis_loc = mainFunc.getLoc();

    axisAnalysis = ShardyLogicalAxisAnalysis(mainFunc);

    if (dumpLogicalAxes) {
      dumpLogicalAxesForMainBlock(mainBlock, axisAnalysis);
    }

    // Step 1: convert main to a distributed.function with indexed shardings.
    Operation *mainScopeOp = mainFunc;
    if (failed(convertMainToDistributedFunction(module_op, mainFunc, mainBlock,
                                                mainScopeOp))) {
      signalPassFailure();
      return;
    }

    // Step 2: remove existing reshards so we can rebuild them uniformly.
    removeExistingReshards(mainScopeOp);

    // Step 3: collect all use-site sharding conflicts and reduction needs.
    std::vector<ShardConflict> conflicts = collectShardConflicts(mainBlock);

    // Step 4: materialize reduction collectives in producer layout.
    if (failed(materializeCollectivesForReductions(module_op, conflicts))) {
      signalPassFailure();
      return;
    }

    // Step 5: materialize layout collectives for remaining conflicts.
    if (failed(materializeCollectivesForConflicts(module_op, conflicts))) {
      signalPassFailure();
      return;
    }

    // Step 6: run clustering to group ops into kernels
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
