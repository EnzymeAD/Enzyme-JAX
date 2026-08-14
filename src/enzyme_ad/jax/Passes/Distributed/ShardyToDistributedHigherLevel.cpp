#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
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
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include <cctype>
#include <optional>
#include <string>
#include <utility>

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

using TensorPartitioningAxes =
    ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis;

static Type getLocalTypeForValue(
    Value value,
    llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
  auto rankedType = dyn_cast<RankedTensorType>(value.getType());
  if (!rankedType) {
    return value.getType();
  }
  return toLocalType(rankedType, partitioningAxes);
}

static std::optional<TensorPartitioningAxes>
getPartitioningForValue(Value value, ShardyLogicalAxisAnalysis &axisAnalysis) {
  if (auto result = dyn_cast<OpResult>(value)) {
    return axisAnalysis.getTensorPartitionDims(result);
  }
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    return axisAnalysis.getTensorPartitionDims(arg);
  }
  return std::nullopt;
}

static std::optional<TensorPartitioningAxes>
getPartitioningForValueOrCastNeighborhood(
    Value value, ShardyLogicalAxisAnalysis &axisAnalysis) {
  if (auto partitioning = getPartitioningForValue(value, axisAnalysis)) {
    return partitioning;
  }

  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>();
      castOp && castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
    if (auto partitioning =
            getPartitioningForValue(castOp.getOperand(0), axisAnalysis)) {
      return partitioning;
    }
  }

  for (OpOperand &use : value.getUses()) {
    auto castUser = dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
    if (!castUser || castUser.getNumOperands() != 1 ||
        castUser.getNumResults() != 1) {
      continue;
    }
    if (auto partitioning =
            getPartitioningForValue(castUser.getResult(0), axisAnalysis)) {
      return partitioning;
    }
  }

  return std::nullopt;
}

static Value
chooseKernelInputOperandValue(Value input,
                              ShardyLogicalAxisAnalysis &axisAnalysis) {
  Value current = input;
  for (int step = 0; step < 8; ++step) {
    auto castOp = current.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1 ||
        castOp.getNumResults() != 1) {
      break;
    }

    Value source = castOp.getOperand(0);
    auto currentPartitioning = getPartitioningForValue(current, axisAnalysis);
    auto sourcePartitioning = getPartitioningForValue(source, axisAnalysis);

    if (sourcePartitioning && !currentPartitioning) {
      current = source;
      continue;
    }

    auto currentType = dyn_cast<RankedTensorType>(current.getType());
    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!currentType || !sourceType ||
        currentType.getRank() != sourceType.getRank()) {
      break;
    }

    bool sourceLooksMoreLocal = false;
    bool comparableShape = true;
    for (int64_t dim = 0; dim < currentType.getRank(); ++dim) {
      if (currentType.isDynamicDim(dim) || sourceType.isDynamicDim(dim)) {
        comparableShape = false;
        break;
      }
      int64_t currentSize = currentType.getDimSize(dim);
      int64_t sourceSize = sourceType.getDimSize(dim);
      if (sourceSize > currentSize) {
        comparableShape = false;
        break;
      }
      if (sourceSize < currentSize) {
        sourceLooksMoreLocal = true;
      }
    }

    if (!comparableShape || !sourceLooksMoreLocal) {
      break;
    }

    current = source;
  }

  return current;
}

static IndexedTensorShardingAttr buildDefaultShardingForType(MLIRContext *ctx,
                                                             Type type) {
  auto emptyAxes = DenseI64ArrayAttr::get(ctx, llvm::ArrayRef<int64_t>{});
  SmallVector<DenseI64ArrayAttr> dimPartitioningAxes;
  if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
    dimPartitioningAxes.reserve(rankedType.getRank());
    for (int64_t i = 0; i < rankedType.getRank(); ++i) {
      dimPartitioningAxes.push_back(emptyAxes);
    }
  }
  return IndexedTensorShardingAttr::get(ctx, dimPartitioningAxes, emptyAxes);
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

      // The return doesn't have its own partitioning mapping,
      // so we look at the producer value.
      std::optional<TensorPartitioningAxes> maybePartitioning = std::nullopt;
      if (OpResult result = dyn_cast<OpResult>(operand.get())) {
        maybePartitioning = axisAnalysis.getTensorPartitionDims(result);
      } else if (BlockArgument arg = dyn_cast<BlockArgument>(operand.get())) {
        maybePartitioning = axisAnalysis.getTensorPartitionDims(arg);
      }
      if (!maybePartitioning) {
        mainFunc.emitError()
            << "missing partitioning mapping for return operand "
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

    auto argShardingsAttr =
        IndexedTensorShardingPerValueAttr::get(ctx, argumentShardings);
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
    auto yieldOp = builder.create<DistributedYieldOp>(
        movedReturnOp.getLoc(), movedReturnOp.getOperands());
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
          auto is_return = isa<func::ReturnOp, distributed::DistributedYieldOp>(
              use.getOwner());
          if (!maybeConsumerSharded && !is_return) {
            use.getOwner()->emitRemark(
                "Found non-sharded use of result number ")
                << result.getResultNumber() << " of op " << op;
            continue;
          }
          if (!is_return && maybeProducerSharded != maybeConsumerSharded) {
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

  LogicalResult clusterOpsIntoKernels(Block *mainBlock,
                                      PartialOrder<Operation *> &order,
                                      ShardyLogicalAxisAnalysis &axisAnalysis) {
    // Want to cluster ops into communication-free distributed.kernels.
    // Need to color the ops so that:
    // 1. No two ops in the same color have a dependency between them
    // outside the color
    // 2. No two ops in the same color have different partitioning axes
    // 3. No communication is created within a kernel regardless of axis
    // assignments.
    // Preconditions:
    // - All collectives have been materialized on local types
    // - may or may not have unrealized type casts between operations
    //   on global values and collectives on local values.
    // Algorithm:
    // - Iterate ops in-order
    // - For each op, if it is not colored, create a new color.
    // - For the new color maintain a set of external input producer ops
    //   and external output consumer ops.
    // - When we add a new op to a color, enque all of its producers / consumers
    //   to see if they are compatible. When dequing, a producer is compatible
    //   if:
    //    - it is not already colored.
    //    - it has the same partitioning axes as the color.
    //    - it is not topologically before any external producer of the color or
    //      topologically after any external consumer of the color.
    // - When the queue runs out, close the color.
    // - When all ops are colored (reach the end of our iteration), for each
    //   color (in order of creation, which we should be able to prove is a
    //   valid topological order), create a distributed.kernel op and move the
    //   ops into it. The input and return types of the kernel should be the
    //   local types, allowing us to get rid of the unrealized casts, while the
    //   block args and returned values should be the global pre-sharded types
    //   from cloning the original global ops.
    using PartitioningAxes =
        ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis;
    const bool debugCluster = dumpLogicalAxes;

    struct ColorState {
      int64_t id;
      PartitioningAxes partitioningAxes;
      llvm::SmallVector<Operation *> members;
      llvm::DenseSet<Operation *> externalProducers;
      llvm::DenseSet<Operation *> externalConsumers;
    };

    auto samePartitioningAxes = [](const PartitioningAxes &lhs,
                                   const PartitioningAxes &rhs) {
      return lhs == rhs;
    };

    auto logAxes = [&](llvm::raw_ostream &os, const PartitioningAxes &axes) {
      os << "[";
      for (const auto &[axisIdx, axis] : llvm::enumerate(axes)) {
        if (axisIdx != 0) {
          os << ", ";
        }
        os << "{";
        for (const auto &[symIdx, symbol] : llvm::enumerate(axis)) {
          if (symIdx != 0) {
            os << " ";
          }
          os << "a" << symbol.getId() << ":" << symbol.getExtent();
        }
        os << "}";
      }
      os << "]";
    };

    auto logOp = [&](StringRef tag, Operation *op) {
      if (!debugCluster) {
        return;
      }
      llvm::errs() << "[cluster] " << tag << ": ";
      if (!op) {
        llvm::errs() << "<null>\n";
        return;
      }
      llvm::errs() << op->getName() << " @" << op << "\n";
    };

    auto isClusterableOp = [](Operation *op) {
      // Keep communication and control/meta ops outside kernels.
      if (isa<DistributedCollectiveOp, DistributedAwait, DistributedYieldOp,
              DistributedKernelOp, UnrealizedConversionCastOp>(op)) {
        return false;
      }

      // Axis dialect ops are sharding metadata plumbing, not computation.
      if (op->getName().getDialectNamespace() == "axis") {
        return false;
      }

      return true;
    };

    auto enqueueNeighbor = [](llvm::SmallVector<Operation *> &queue,
                              Operation *op) {
      if (op) {
        queue.push_back(op);
      }
    };

    auto tryAcceptCandidate =
        [&](Operation *candidate, ColorState &color,
            llvm::DenseMap<Operation *, int64_t> &opToColor) {
          if (!candidate || !isClusterableOp(candidate) ||
              opToColor.contains(candidate)) {
            if (debugCluster) {
              llvm::errs() << "[cluster] reject early in color " << color.id
                           << " candidate=";
              if (candidate) {
                llvm::errs() << candidate->getName();
              } else {
                llvm::errs() << "<null>";
              }
              llvm::errs() << "\n";
            }
            return false;
          }

          PartitioningAxes candidateAxes =
              axisAnalysis.getPartitioningAxes(candidate);
          if (!samePartitioningAxes(candidateAxes, color.partitioningAxes)) {
            if (debugCluster) {
              llvm::errs() << "[cluster] reject axis mismatch in color "
                           << color.id << " candidate=" << candidate->getName()
                           << " axes=";
              logAxes(llvm::errs(), candidateAxes);
              llvm::errs() << " expected=";
              logAxes(llvm::errs(), color.partitioningAxes);
              llvm::errs() << "\n";
            }
            return false;
          }

          // Respect color boundary topology constraints.
          for (Operation *externalProducer : color.externalProducers) {
            if (order.compare(candidate, externalProducer) == Order::LessThan) {
              if (debugCluster) {
                llvm::errs()
                    << "[cluster] reject producer boundary in color "
                    << color.id << " candidate=" << candidate->getName()
                    << " producer=" << externalProducer->getName() << "\n";
              }
              return false;
            }
          }
          for (Operation *externalConsumer : color.externalConsumers) {
            if (order.compare(candidate, externalConsumer) ==
                Order::GreaterThan) {
              if (debugCluster) {
                llvm::errs()
                    << "[cluster] reject consumer boundary in color "
                    << color.id << " candidate=" << candidate->getName()
                    << " consumer=" << externalConsumer->getName() << "\n";
              }
              return false;
            }
          }

          opToColor[candidate] = color.id;
          color.members.push_back(candidate);
          logOp(("accept color " + llvm::Twine(color.id)).str(), candidate);
          return true;
        };

    auto updateBoundaryAndQueue =
        [&](Operation *accepted, ColorState &color,
            llvm::DenseMap<Operation *, int64_t> &opToColor,
            llvm::SmallVector<Operation *> &queue) {
          for (OpOperand &operand : accepted->getOpOperands()) {
            // don't need to color block args but need to consider
            // dependencies through them
            if (auto blockArg = dyn_cast<BlockArgument>(operand.get());
                blockArg && blockArg.getParentBlock() == mainBlock) {
              for (OpOperand &otherUse : blockArg.getUses()) {
                Operation *otherUser = otherUse.getOwner();
                if (!otherUser || otherUser->getBlock() != mainBlock ||
                    opToColor.contains(otherUser)) {
                  continue;
                }
                enqueueNeighbor(queue, otherUser);
              }
            }

            Operation *producer = operand.get().getDefiningOp();
            if (!producer || producer->getBlock() != mainBlock) {
              continue;
            }
            if (!opToColor.contains(producer)) {
              color.externalProducers.insert(producer);
              logOp(("boundary producer color " + llvm::Twine(color.id)).str(),
                    producer);
              enqueueNeighbor(queue, producer);
            }
          }

          for (Value result : accepted->getResults()) {
            for (OpOperand &use : result.getUses()) {
              Operation *consumer = use.getOwner();
              if (!consumer || consumer->getBlock() != mainBlock) {
                continue;
              }
              if (!opToColor.contains(consumer)) {
                color.externalConsumers.insert(consumer);
                logOp(
                    ("boundary consumer color " + llvm::Twine(color.id)).str(),
                    consumer);
                enqueueNeighbor(queue, consumer);
              }
            }
          }
        };

    llvm::DenseMap<Operation *, int64_t> opToColor;
    llvm::SmallVector<ColorState> colors;
    int64_t nextColorId = 0;

    for (Operation &opRef : mainBlock->getOperations()) {
      Operation *seed = &opRef;
      if (!isClusterableOp(seed) || opToColor.contains(seed)) {
        continue;
      }

      logOp("seed", seed);

      PartitioningAxes seedAxes = axisAnalysis.getPartitioningAxes(seed);
      ColorState color{nextColorId++, std::move(seedAxes), {}, {}, {}};
      if (debugCluster) {
        llvm::errs() << "[cluster] create color " << color.id << " axes=";
        logAxes(llvm::errs(), color.partitioningAxes);
        llvm::errs() << "\n";
      }

      // Seed the color and then grow by producer/consumer exploration.
      (void)tryAcceptCandidate(seed, color, opToColor);
      llvm::SmallVector<Operation *> queue;
      updateBoundaryAndQueue(seed, color, opToColor, queue);

      for (size_t queueIdx = 0; queueIdx < queue.size(); ++queueIdx) {
        Operation *candidate = queue[queueIdx];
        logOp(("dequeue color " + llvm::Twine(color.id)).str(), candidate);
        if (!tryAcceptCandidate(candidate, color, opToColor)) {
          continue;
        }
        updateBoundaryAndQueue(candidate, color, opToColor, queue);
      }

      if (debugCluster) {
        llvm::errs() << "[cluster] close color " << color.id
                     << " members=" << color.members.size()
                     << " ext_prod=" << color.externalProducers.size()
                     << " ext_cons=" << color.externalConsumers.size() << "\n";
      }

      colors.push_back(std::move(color));
    }

    if (debugCluster) {
      llvm::errs() << "[cluster] total colors: " << colors.size() << "\n";
    }

    auto *ctx = &getContext();

    // NOTE: Kernel boundary values that are ranked tensors are expected to
    // carry sharding information. We recover this from direct analysis,
    // representative use-sites, and nearby unrealized casts; if still missing,
    // treat it as an invariant violation for compute values. We only allow
    // default-empty sharding for non-ranked values.
    OpBuilder builder(ctx);
    for (size_t colorIdx = 0; colorIdx < colors.size(); ++colorIdx) {
      const ColorState &color = colors[colorIdx];
      if (color.members.empty()) {
        continue;
      }

      llvm::DenseSet<Operation *> memberSet(color.members.begin(),
                                            color.members.end());

      llvm::SmallVector<Operation *> orderedMembers;
      orderedMembers.reserve(color.members.size());
      for (Operation &op : mainBlock->getOperations()) {
        if (memberSet.contains(&op)) {
          orderedMembers.push_back(&op);
        }
      }
      if (orderedMembers.empty()) {
        continue;
      }

      llvm::DenseMap<Value, int64_t> inputIndices;
      llvm::SmallVector<Value> kernelInputs;
      // with casts and collectives inserted the producer may not be known to
      // the logical axis analysis for a value, but the use should be, allowing
      // us to look up partitioning axes.
      llvm::DenseMap<Value, OpOperand *> representativeInputUse;
      llvm::DenseMap<Value, int64_t> outputIndices;
      llvm::SmallVector<Value> kernelOutputs;

      for (Operation *member : orderedMembers) {
        for (OpOperand &operand : member->getOpOperands()) {
          Value input = operand.get();
          Operation *producer = input.getDefiningOp();
          bool producedInsideColor = producer &&
                                     producer->getBlock() == mainBlock &&
                                     memberSet.contains(producer);
          if (producedInsideColor) {
            continue;
          }
          if (!inputIndices.contains(input)) {
            inputIndices[input] = static_cast<int64_t>(kernelInputs.size());
            kernelInputs.push_back(input);
            representativeInputUse[input] = &operand;
          }
        }
      }

      for (Operation *member : orderedMembers) {
        for (Value result : member->getResults()) {
          bool escapesColor = false;
          for (OpOperand &use : result.getUses()) {
            Operation *consumer = use.getOwner();
            bool consumedInsideColor = consumer &&
                                       consumer->getBlock() == mainBlock &&
                                       memberSet.contains(consumer);
            if (!consumedInsideColor) {
              escapesColor = true;
              break;
            }
          }
          if (!escapesColor) {
            continue;
          }
          if (!outputIndices.contains(result)) {
            outputIndices[result] = static_cast<int64_t>(kernelOutputs.size());
            kernelOutputs.push_back(result);
          }
        }
      }

      SmallVector<Value> kernelInputOperands;
      kernelInputOperands.reserve(kernelInputs.size());
      SmallVector<Type> kernelBlockArgTypes;
      kernelBlockArgTypes.reserve(kernelInputs.size());

      llvm::DenseMap<AxisSymbol, int64_t> symbolToPartitioningAxisIdx;
      SmallVector<IndexedTensorShardingAttr> inputShardings;
      inputShardings.reserve(kernelInputs.size());

      Operation *insertBefore = mainBlock->getTerminator();
      builder.setInsertionPoint(insertBefore);

      for (Value input : kernelInputs) {
        kernelBlockArgTypes.push_back(input.getType());

        auto maybePartitioning =
            axisAnalysis.getTensorPartitionDims(*representativeInputUse[input]);
        if (!maybePartitioning) {
          maybePartitioning =
              getPartitioningForValueOrCastNeighborhood(input, axisAnalysis);
        }

        if (!maybePartitioning && isa<RankedTensorType>(input.getType())) {
          insertBefore->emitError()
              << "missing sharding for ranked kernel input value " << input;
          return failure();
        }

        Type localInputType = input.getType();
        if (maybePartitioning) {
          localInputType = getLocalTypeForValue(input, *maybePartitioning);
        }

        Value localInputValue =
            chooseKernelInputOperandValue(input, axisAnalysis);
        if (localInputValue.getType() != localInputType) {
          localInputValue =
              builder
                  .create<UnrealizedConversionCastOp>(
                      insertBefore->getLoc(), localInputType, localInputValue)
                  .getResult(0);
        }
        kernelInputOperands.push_back(localInputValue);

        if (auto rankedType = dyn_cast<RankedTensorType>(localInputType);
            rankedType && maybePartitioning) {
          inputShardings.push_back(buildIndexedShardingAttr(
              rankedType, *maybePartitioning, symbolToPartitioningAxisIdx));
        } else {
          inputShardings.push_back(
              buildDefaultShardingForType(ctx, localInputType));
        }
      }

      SmallVector<IndexedTensorShardingAttr> outputShardings;
      outputShardings.reserve(kernelOutputs.size());
      SmallVector<Type> kernelResultTypes;
      kernelResultTypes.reserve(kernelOutputs.size());
      for (Value output : kernelOutputs) {
        auto maybePartitioning =
            getPartitioningForValueOrCastNeighborhood(output, axisAnalysis);
        if (!maybePartitioning && isa<RankedTensorType>(output.getType())) {
          insertBefore->emitError()
              << "missing sharding for ranked kernel output value " << output;
          return failure();
        }
        // Kernel returns are in local type, global type recoverable
        // from yield or from multiplying the local type by the sharding.
        Type globalOutputType = output.getType();
        Type localOutputType = getLocalTypeForValue(output, *maybePartitioning);
        kernelResultTypes.push_back(localOutputType);

        if (auto rankedType = dyn_cast<RankedTensorType>(globalOutputType);
            rankedType && maybePartitioning) {
          outputShardings.push_back(buildIndexedShardingAttr(
              rankedType, *maybePartitioning, symbolToPartitioningAxisIdx));
        } else {
          outputShardings.push_back(
              buildDefaultShardingForType(ctx, globalOutputType));
        }
      }

      SmallVector<Value> kernelPartitioningAxes(
          symbolToPartitioningAxisIdx.size());
      for (const auto &[symbol, idx] : symbolToPartitioningAxisIdx) {
        kernelPartitioningAxes[idx] = getOrCreatePartitioningAxisGroup(symbol);
      }

      auto inputShardingsAttr =
          IndexedTensorShardingPerValueAttr::get(ctx, inputShardings);
      auto outputShardingsAttr =
          IndexedTensorShardingPerValueAttr::get(ctx, outputShardings);

      auto kernel = builder.create<DistributedKernelOp>(
          insertBefore->getLoc(), TypeRange(kernelResultTypes),
          ValueRange(kernelInputOperands), ValueRange(kernelPartitioningAxes),
          inputShardingsAttr, outputShardingsAttr);

      Region &kernelBody = kernel->getRegion(0);
      if (kernelBody.empty()) {
        kernelBody.push_back(new Block());
      }
      Block &kernelBlock = kernelBody.front();
      for (Type argType : kernelBlockArgTypes) {
        kernelBlock.addArgument(argType, insertBefore->getLoc());
      }

      IRMapping mapping;
      for (auto [idx, input] : llvm::enumerate(kernelInputs)) {
        mapping.map(input, kernelBlock.getArgument(idx));
      }

      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&kernelBlock);
      for (Operation *member : orderedMembers) {
        bodyBuilder.clone(*member, mapping);
      }

      SmallVector<Value> yieldedValues;
      yieldedValues.reserve(kernelOutputs.size());
      for (Value output : kernelOutputs) {
        yieldedValues.push_back(mapping.lookup(output));
      }
      bodyBuilder.setInsertionPointToEnd(&kernelBlock);
      bodyBuilder.create<DistributedYieldOp>(insertBefore->getLoc(),
                                             yieldedValues);

      for (auto [resultIdx, oldValue] : llvm::enumerate(kernelOutputs)) {
        Value newValue = kernel.getResult(resultIdx);
        for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
          Operation *consumer = use.getOwner();
          bool consumedInsideColor = consumer &&
                                     consumer->getBlock() == mainBlock &&
                                     memberSet.contains(consumer);
          if (!consumedInsideColor) {
            rewriteUseWithValue(builder, insertBefore->getLoc(), &use,
                                newValue);
          }
        }
      }

      for (auto it = orderedMembers.rbegin(); it != orderedMembers.rend();
           ++it) {
        (*it)->erase();
      }

      if (debugCluster) {
        llvm::errs() << "[cluster] materialized kernel color=" << color.id
                     << " members=" << orderedMembers.size()
                     << " inputs=" << kernelInputs.size()
                     << " outputs=" << kernelOutputs.size() << "\n";
      }
    }

    if (!mlir::sortTopologically(mainBlock)) {
      mainBlock->getParentOp()->emitError()
          << "failed to reorder block operations topologically";
      return failure();
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();

    const auto &mainFunctionAnalysis = getAnalysis<FindMainFunctionAnalysis>();
    if (!mainFunctionAnalysis.isValid()) {
      if (!mainFunctionAnalysis.hasMainFunction()) {
        emitWarning(module_op.getLoc())
            << "no main function found; skipping pass";
        return;
      }
      emitError(module_op.getLoc())
          << "multiple symbols named 'main' found across func.func and "
             "distributed.DistributedFunction";
      signalPassFailure();
      return;
    }

    func::FuncOp mainFunc = mainFunctionAnalysis.getMainFuncOp();
    if (!mainFunc) {
      emitWarning(module_op.getLoc())
          << "main is not a func.func; skipping pass";
      return;
    }

    Block *mainBlock = mainFunctionAnalysis.getMainBlock();
    if (!mainBlock) {
      emitError(mainFunc.getLoc()) << "main function has no body";
      signalPassFailure();
      return;
    }

    // prep for building axes
    axis_builder = OpBuilder(module_op.getContext());
    axis_builder->setInsertionPointToStart(&module_op.getBodyRegion().front());
    axis_loc = mainFunc.getLoc();

    const auto &mainAxisAnalysis =
        getAnalysis<MainFunctionShardyLogicalAxisAnalysis>();
    if (!mainAxisAnalysis.isValid()) {
      emitError(mainFunc.getLoc())
          << "failed to build module-scoped main logical axis analysis";
      signalPassFailure();
      return;
    }
    axisAnalysis = mainAxisAnalysis.getAnalysis();

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
    auto &order_analysis =
        getAnalysis<MainFunctionSSABlockPartialOrderAnalysis>();
    if (!order_analysis.isValid()) {
      emitError(module_op.getLoc())
          << "failed to build module-scoped main SSA partial order analysis";
      signalPassFailure();
      return;
    }
    auto &order = order_analysis.getPartialOrder();
    if (failed(clusterOpsIntoKernels(mainBlock, order, axisAnalysis))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
