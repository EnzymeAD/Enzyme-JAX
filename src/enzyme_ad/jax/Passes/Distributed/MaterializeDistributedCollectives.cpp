#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include <optional>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_MATERIALIZEDISTRIBUTEDCOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

using TV_AxisFactor = mlir::TypedValue<mlir::enzyme::axis::AxisFactorType>;
using TV_Axis = mlir::TypedValue<mlir::enzyme::axis::AxisTypeInterface>;
using TV_FactorGroup = mlir::TypedValue<mlir::enzyme::axis::FactorGroupType>;

// Converts typed SSA ranges to Value ranges for axis helper builders.
template <typename RangeT>
static llvm::SmallVector<Value> asValues(const RangeT &groups) {
  llvm::SmallVector<Value> values;
  values.reserve(groups.size());
  for (Value group : groups) {
    values.push_back(group);
  }
  return values;
}

// Removes explicit sdy.reshard ops after we have committed to rebuilding them.
static void removeExistingReshards(Operation *scopeOp) {
  std::vector<sdy::ReshardOp> toRemove;
  scopeOp->walk([&](sdy::ReshardOp reshardOp) {
    toRemove.push_back(reshardOp);
    reshardOp.getResult().replaceAllUsesWith(reshardOp.getInput());
  });
  for (sdy::ReshardOp reshardOp : toRemove) {
    reshardOp.erase();
  }
}

// Computes producer-local tensor type by dividing each dim by partition extent.
static RankedTensorType
toLocalType(RankedTensorType globalType,
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
  return RankedTensorType::get(localShape, globalType.getElementType());
}

struct MaterializeDistributedCollectivesPass
    : public impl::MaterializeDistributedCollectivesPassBase<
          MaterializeDistributedCollectivesPass> {
  using MaterializeDistributedCollectivesPassBase::
      MaterializeDistributedCollectivesPassBase;

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

  llvm::DenseMap<AxisSymbol, TV_AxisFactor> symbolToLogicalAxis;
  ShardyLogicalAxisAnalysis axisAnalysis;
  std::optional<OpBuilder> axisBuilder;
  std::optional<Location> axisLoc;

  // Lazily materializes one LogicalMeshAxesOp value for each logical symbol.
  TV_AxisFactor getOrCreateLogicalAxisForSymbol(AxisSymbol symbol) {
    auto it = symbolToLogicalAxis.find(symbol);
    if (it != symbolToLogicalAxis.end()) {
      return it->second;
    }

    auto op = axisBuilder->create<mlir::enzyme::distributed::LogicalMeshAxesOp>(
        *axisLoc, symbol.getExtent());
    auto asFactor =
        axis::viewAxesAsFactors(op.getAxes(), *axisBuilder, *axisLoc);
    symbolToLogicalAxis[symbol] = asFactor.front();
    return asFactor.front();
  }

  // Collects logical-axis factors for one tensor partitioning dimension list.
  llvm::SmallVector<TV_AxisFactor>
  getLogicalAxesForSymbols(llvm::ArrayRef<AxisSymbol> symbols) {
    llvm::SmallVector<TV_AxisFactor> logicalAxes;
    logicalAxes.reserve(symbols.size());
    for (const auto &symbol : symbols) {
      logicalAxes.push_back(getOrCreateLogicalAxisForSymbol(symbol));
    }
    return logicalAxes;
  }

  // Builds one positional factor group per tensor dimension for a tensor view
  // cast. Each group preserves the logical-axis provenance of that dimension.
  SmallVector<Value> getTensorPartitioningAxisGroups(
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    SmallVector<Value> partitioningAxisGroups;
    partitioningAxisGroups.reserve(partitioningAxes.size());
    for (const auto &dimensionAxes : partitioningAxes) {
      auto factors = getLogicalAxesForSymbols(dimensionAxes);
      partitioningAxisGroups.push_back(axis::viewFactorsAsProduct(
          asValues(factors), *axisBuilder, *axisLoc));
    }
    return partitioningAxisGroups;
  }

  // Reuse distributed.function's serialized logical-axis factors so every
  // materialized cast and collective shares its existing SSA axis anchors.
  LogicalResult seedLogicalAxesFromFunction(DistributedFunctionOp function) {
    auto partitioningAxes = function.getPartitioningAxes();
    auto seedValue = [&](Value value, IndexedTensorShardingAttr sharding) {
      if (!isa<RankedTensorType>(value.getType())) {
        return success();
      }

      std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
          maybePartitioning;
      if (auto result = dyn_cast<OpResult>(value)) {
        maybePartitioning = axisAnalysis.getTensorPartitionDims(result);
      } else if (auto argument = dyn_cast<BlockArgument>(value)) {
        maybePartitioning = axisAnalysis.getTensorPartitionDims(argument);
      }
      if (!maybePartitioning || maybePartitioning->size() !=
                                    sharding.getDimPartitioningAxes().size()) {
        function.emitError()
            << "failed to align function sharding metadata with logical axes";
        return failure();
      }

      for (auto [symbols, axisIndices] : llvm::zip_equal(
               *maybePartitioning, sharding.getDimPartitioningAxes())) {
        if (symbols.size() != axisIndices.size()) {
          function.emitError()
              << "function sharding metadata does not match logical-axis "
                 "basis";
          return failure();
        }
        for (auto [symbol, axisIndex] :
             llvm::zip_equal(symbols, axisIndices.asArrayRef())) {
          if (axisIndex < 0 ||
              axisIndex >= static_cast<int64_t>(partitioningAxes.size())) {
            function.emitError()
                << "function sharding metadata references an invalid axis";
            return failure();
          }
          auto factorGroup =
              dyn_cast<TV_FactorGroup>(partitioningAxes[axisIndex]);
          if (!factorGroup) {
            function.emitError()
                << "function sharding metadata references a non-factor axis";
            return failure();
          }
          auto factors = axis::getProductProvenanceFactors(factorGroup);
          if (failed(factors) || factors->size() != 1) {
            function.emitError()
                << "function partitioning axis must have one logical factor";
            return failure();
          }
          TV_AxisFactor factor = factors->front();
          if (failed(axisAnalysis.assignLogicalAxis(symbol, factor))) {
            return failure();
          }
          auto [it, inserted] =
              symbolToLogicalAxis.try_emplace(symbol, factor);
          if (!inserted && it->second != factor) {
            function.emitError()
                << "unified logical axis has distinct SSA factor anchors";
            return failure();
          }
        }
      }
      return success();
    };

    for (auto [argument, sharding] :
         llvm::zip_equal(function.getBody().front().getArguments(),
                         function.getArgumentShardings().getShardings())) {
      if (failed(seedValue(argument, sharding))) {
        return failure();
      }
    }

    auto yield = dyn_cast<DistributedYieldOp>(
        function.getBody().front().getTerminator());
    if (!yield || yield.getNumOperands() !=
                      function.getOutputShardings().getShardings().size()) {
      function.emitError() << "failed to align function results with sharding "
                              "metadata";
      return failure();
    }
    for (auto [operand, sharding] :
         llvm::zip_equal(yield.getOperands(),
                         function.getOutputShardings().getShardings())) {
      if (failed(seedValue(operand, sharding))) {
        return failure();
      }
    }
    return success();
  }

  // Builds a local-type-aware axis product for collective mapping operands.
  TV_FactorGroup toLocallyTypedAxisProduct(
      RankedTensorType localType,
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    assert(localType.getRank() == partitioningAxes.size() &&
           "local type and partitioning axes must have the same rank");
    llvm::SmallVector<TV_AxisFactor> factors;
    for (auto [axisIndex, axisSymbols] : llvm::enumerate(partitioningAxes)) {
      assert(!axisSymbols.empty() && "partitioning axes must be non-empty");
      auto logicalAxes = getLogicalAxesForSymbols(axisSymbols);
      factors.append(logicalAxes.begin(), logicalAxes.end());

      auto localAxis = axisBuilder->create<mlir::enzyme::axis::AxisGetAxisOp>(
          *axisLoc, localType, axisIndex);
      llvm::SmallVector<TV_Axis> localAxes = {localAxis.getAxis()};
      auto localAxisFactor =
          axis::viewAxesAsFactors(localAxes, *axisBuilder, *axisLoc);
      factors.push_back(localAxisFactor.front());
    }

    auto factorGroup = axisBuilder->create<mlir::enzyme::axis::AxisProductOp>(
        *axisLoc, asValues(factors));
    return factorGroup.getProduct();
  }

  // Computes mesh factor-group operands from partitioning symbols.
  TV_FactorGroup getMeshForTensorPartitioning(
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    llvm::SmallVector<llvm::SmallVector<TV_AxisFactor>> logicalAxes;
    logicalAxes.reserve(partitioningAxes.size());
    for (const auto &axis : partitioningAxes) {
      logicalAxes.push_back(getLogicalAxesForSymbols(axis));
    }
    auto product = axisBuilder->create<mlir::enzyme::axis::AxisProductOp>(
        *axisLoc, asValues(flattenNested(logicalAxes)));
    return product.getProduct();
  }

  // Scans SSA uses to find sharding conflicts and reduction requirements.
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
          auto isReturn = isa<func::ReturnOp, distributed::DistributedYieldOp>(
              use.getOwner());
          if (!maybeConsumerSharded && !isReturn) {
            use.getOwner()->emitRemark(
                "Found non-sharded use of result number ")
                << result.getResultNumber() << " of op " << op;
            continue;
          }
          if (!isReturn && maybeProducerSharded != maybeConsumerSharded) {
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

  // Rewrites one operand use, inserting casts when type adaptation is required.
  void rewriteUseWithValue(
      OpBuilder &builder, Location loc, OpOperand *use, Value replacement,
      llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
    Type expectedUseType = use->get().getType();
    Value valueForUse = replacement;
    if (valueForUse.getType() != expectedUseType) {
      builder.setInsertionPoint(use->getOwner());
      if (isa<RankedTensorType>(valueForUse.getType()) &&
          isa<RankedTensorType>(expectedUseType)) {
        auto partitioningAxisGroups =
            getTensorPartitioningAxisGroups(partitioningAxes);
        valueForUse =
            builder
                .create<DistributedCastLocalToGlobalOp>(
                    loc, expectedUseType, valueForUse, partitioningAxisGroups)
                .getOutput();
      } else {
        valueForUse = builder
                          .create<UnrealizedConversionCastOp>(
                              loc, expectedUseType, valueForUse)
                          .getResult(0);
      }
    }
    use->set(valueForUse);
  }

  // Builds one distributed collective+await pair, cloning reduction regions.
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

  // Materializes producer-layout reductions and updates conflict state.
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
      auto partitioningAxisGroups =
          getTensorPartitioningAxisGroups(conflict.producerPartitioningAxes);
      auto localCast = builder.create<DistributedCastGlobalToLocalOp>(
          conflict.value.getLoc(), localType, conflict.value,
          partitioningAxisGroups);
      conflict.currentValue = localCast.getOutput();
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
                            conflict.currentValue,
                            conflict.producerPartitioningAxes);
      }
    }

    return success();
  }

  // Materializes layout-only collectives for use-sites that still conflict.
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
                            conflict.currentValue,
                            conflict.producerPartitioningAxes);
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
        rewriteUseWithValue(builder, conflict.value.getLoc(), use, collective,
                            *rhsPartitioningAxes);
      }
    }

    return success();
  }

  // Replaces reshards with collectives for the current main function-like op.
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    const auto &mainFunctionAnalysis = getAnalysis<FindMainFunctionAnalysis>();
    if (!mainFunctionAnalysis.isValid()) {
      if (!mainFunctionAnalysis.hasMainFunction()) {
        emitWarning(moduleOp.getLoc())
            << "no main function found; skipping pass";
        return;
      }
      emitError(moduleOp.getLoc()) << "failed to find main function";
      signalPassFailure();
      return;
    }

    Operation *mainScopeOp = mainFunctionAnalysis.getMainFunctionOp();
    if (!mainScopeOp) {
      emitWarning(moduleOp.getLoc())
          << "main is not a function-like op; skipping pass";
      return;
    }

    Block *mainBlock = mainFunctionAnalysis.getMainBlock();
    if (!mainBlock) {
      mainScopeOp->emitRemark()
          << "main is not a single-block function-like op; skipping pass";
      signalPassFailure();
      return;
    }

    axisBuilder = OpBuilder(moduleOp.getContext());
    axisBuilder->setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    axisLoc = mainScopeOp->getLoc();

    const auto &mainAxisAnalysis =
        getAnalysis<MainFunctionShardyLogicalAxisAnalysis>();
    if (!mainAxisAnalysis.isValid()) {
      mainScopeOp->emitRemark()
          << "failed to build module-scoped main logical axis analysis";
      signalPassFailure();
      return;
    }
    axisAnalysis = mainAxisAnalysis.getAnalysis();

    auto distributedFunction = dyn_cast<DistributedFunctionOp>(mainScopeOp);
    if (!distributedFunction ||
        failed(seedLogicalAxesFromFunction(distributedFunction))) {
      signalPassFailure();
      return;
    }

    removeExistingReshards(mainScopeOp);
    std::vector<ShardConflict> conflicts = collectShardConflicts(mainBlock);

    if (failed(materializeCollectivesForReductions(moduleOp, conflicts))) {
      signalPassFailure();
      return;
    }
    if (failed(materializeCollectivesForConflicts(moduleOp, conflicts))) {
      signalPassFailure();
      return;
    }

    // Explicit tensor view casts retain the logical-axis basis across this
    // rewrite, so clustering may safely reuse the analysis. Preserve both
    // layers because the module wrapper references the child analysis.
    markAnalysesPreserved<ShardyLogicalAxisAnalysis>();
    markAnalysesPreserved<MainFunctionShardyLogicalAxisAnalysis>();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
