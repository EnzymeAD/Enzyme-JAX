#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include <cctype>
#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDHIGHERLEVELPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

template <typename T>
llvm::SmallVector<T>
flatten(const llvm::SmallVector<llvm::SmallVector<T>> &nested) {
  llvm::SmallVector<T> flat;
  for (const auto &inner : nested) {
    flat.append(inner.begin(), inner.end());
  }
  return flat;
}

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

static FlatSymbolRefAttr
ensurePlaceholderAllReduceReductionFunction(ModuleOp moduleOp,
                                            Type elementType) {
  std::string symbolName =
      ("_distributed_addscalar_" + getTypeSuffix(elementType));

  if (!moduleOp.lookupSymbol<func::FuncOp>(symbolName)) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto scalarTensorType = RankedTensorType::get({}, elementType);
    auto reductionFnType = builder.getFunctionType(
        {scalarTensorType, scalarTensorType}, {scalarTensorType});
    auto reductionFn = builder.create<func::FuncOp>(
        moduleOp.getLoc(), symbolName, reductionFnType);
    reductionFn.setPrivate();

    Block *entry = reductionFn.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto add = builder.create<stablehlo::AddOp>(
        moduleOp.getLoc(), entry->getArgument(0), entry->getArgument(1));
    builder.create<func::ReturnOp>(moduleOp.getLoc(), add.getResult());
  }

  return FlatSymbolRefAttr::get(moduleOp.getContext(), symbolName);
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
    llvm::SmallVector<AxisSymbol> reductionAxes;
    llvm::SmallVector<OpOperand *> conflictingUses;
    llvm::SmallVector<OpOperand *> nonConflictingUses;
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

  TV_FactorGroup getOpMesh(Operation *op) {
    // the mesh is just the product of all partitioning axes
    // attatched to the op.
    auto partitioningAxes = axisAnalysis.getPartitioningAxes(op);
    llvm::SmallVector<llvm::SmallVector<TV_AxisFactor>> logical_axes;
    logical_axes.reserve(partitioningAxes.size());
    for (const auto &axis : partitioningAxes) {
      logical_axes.push_back(getLogicalAxesForSymbols(axis));
    }
    auto product = axis_builder->create<mlir::enzyme::axis::AxisProductOp>(
        *axis_loc, asValues(flatten(logical_axes)));
    return product.getProduct();
  }

  // Removes existing `sdy.reshard` ops and forwards their input values.
  // We rebuild reshards as explicit collectives later once all conflicts are
  // known.
  void removeExistingReshards(func::FuncOp mainFunc) {
    std::vector<sdy::ReshardOp> toRemove;
    mainFunc.walk([&](sdy::ReshardOp reshardOp) {
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
        conflict.reductionAxes = axisAnalysis.getReductionAxes(result);
        for (OpOperand &use : result.getUses()) {
          auto maybeConsumerSharded = axisAnalysis.getTensorPartitionDims(use);
          if (!maybeConsumerSharded) {
            use.getOwner()->emitRemark("Found non-sharded use of result number ")
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

  // Materializes collectives for each detected conflict, including local-shape
  // typing and optional reduction setup when reduction axes are present.
  LogicalResult materializeCollectivesForConflicts(
      ModuleOp moduleOp, llvm::ArrayRef<ShardConflict> conflicts) {
    OpBuilder builder(moduleOp.getContext());

    // TODO this can be made more intelligent, either by
    // improving collective creation here or optimizing downstream.
    for (const ShardConflict &conflict : conflicts) {
      // Need to insert unrealized type conversion for global type to local
      // type.
      auto partitioningAxes =
          axisAnalysis.getTensorPartitionDims(conflict.value).value();
      auto originalType = conflict.value.getType();
      RankedTensorType globalType;
      if (!(globalType = dyn_cast<RankedTensorType>(originalType))) {
        conflict.value.getOwner()->emitError(
            "Found non-ranked tensor type for sharded value ")
            << conflict.value;
        return failure();
      }
      auto localType = toLocalType(globalType, partitioningAxes);
      builder.setInsertionPointAfterValue(conflict.value);
      auto unrealizedCast = builder.create<UnrealizedConversionCastOp>(
          conflict.value.getLoc(), localType, conflict.value);
      auto localValue = unrealizedCast.getResult(0);

      auto reductionDims = getLogicalAxesForSymbols(conflict.reductionAxes);
      auto lhsDims = toLocallyTypedAxisProduct(localType, partitioningAxes);
      auto lhsMesh = getOpMesh(conflict.value.getOwner());

      auto createCollective = [&](TV_FactorGroup rhsMesh, TV_FactorGroup rhsDims,
                                  Type collectiveOutputType) {
        llvm::SmallVector<Value> reductionGroupValues;
        llvm::SmallVector<Attribute> reductionFunctions;
        if (!reductionDims.empty()) {
          auto reductionGroup =
              builder
                  .create<mlir::enzyme::axis::AxisProductOp>(
                      conflict.value.getLoc(), asValues(reductionDims))
                  .getProduct();
          reductionGroupValues.push_back(reductionGroup);
          reductionFunctions.push_back(
              ensurePlaceholderAllReduceReductionFunction(
                  moduleOp, globalType.getElementType()));
        }
        auto mapping = builder.create<mlir::enzyme::axis::AxisMapOp>(
            conflict.value.getLoc(), ValueRange{lhsDims}, ValueRange{rhsDims});
        auto collectiveAndAwait =
            mlir::enzyme::distributed::createCollectiveAndAwait(
                builder, conflict.value.getLoc(), localValue, lhsMesh, rhsMesh,
                ValueRange(reductionGroupValues),
                builder.getArrayAttr(reductionFunctions), mapping.getMap(),
                collectiveOutputType);
        return collectiveAndAwait.await.getValue();
      };
      auto rewriteUseWithCollective = [&](OpOperand *use) {
        auto rhsPartitioningAxes = axisAnalysis.getTensorPartitionDims(*use).value();
        auto rhsLocalType = toLocalType(globalType, rhsPartitioningAxes);
        auto rhsDims =
            toLocallyTypedAxisProduct(rhsLocalType, rhsPartitioningAxes);
        auto rhsMesh = getOpMesh(use->getOwner());
        Value collective = createCollective(rhsMesh, rhsDims, rhsLocalType);
        Type expectedUseType = use->get().getType();
        if (collective.getType() != expectedUseType) {
          collective = builder
                           .create<UnrealizedConversionCastOp>(
                               conflict.value.getLoc(), expectedUseType,
                               collective)
                           .getResult(0);
        }
        use->set(collective);
      };
      for (OpOperand *use : conflict.conflictingUses) {
        rewriteUseWithCollective(use);
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

    // After partition axis discovery, we have introduced more axes and possibly discovered some
    // points where we cannot merge axes due to incompatible factorization or
    // other constraints. Therefore, our next step is to:
    // - rewrite any reshards into our collectives
    // - find any other SSA reshards with incompatible factorization and/or
    // any reduction axes and insert a collective. Note: collectives currently
    // work on local tensor shapes, so we will have to insert unrealized type
    // conversions with maximal sharding size. For instance, if global [16 x
    // 4] tensor is sharded over [a(16), --], then the resulting local tensor
    // size is [1 x 4]. Luckily, we can just build the reshards as a single
    // flat map and let canonicalization figure out the separable factors. The
    // LHS and RHS mesh are likely on different axes, so we will need to
    // remember to set hte input/output meshes separately. (Of course, we
    // first need to convert symbols --> logical axes). Then, we can run the
    // clustering pass much easier: we break on unrealized type conversions //
    // reshards, and topology issues.
    // TODO: reduces and reshards at same time or different?

    // Step 1: remove existing reshards so we can rebuild them uniformly.
    removeExistingReshards(mainFunc);

    // Step 2: collect all use-site sharding conflicts and reduction needs.
    std::vector<ShardConflict> conflicts = collectShardConflicts(mainBlock);

    // Step 3: materialize collectives for each conflict.
    if (failed(materializeCollectivesForConflicts(module_op, conflicts))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
