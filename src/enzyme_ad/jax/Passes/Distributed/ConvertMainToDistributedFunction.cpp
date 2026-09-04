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
#include <string>
#include <utility>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_CONVERTMAINTODISTRIBUTEDFUNCTIONPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

using TV_AxisFactor = mlir::TypedValue<mlir::enzyme::axis::AxisFactorType>;
using TV_FactorGroup = mlir::TypedValue<mlir::enzyme::axis::FactorGroupType>;
using TensorPartitioningAxes =
    ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis;

// Can't use dedicated Shardy pass since we need to keep the mesh attributes
// around for the Reshard ops (for now)
static ArrayAttr dropShardyAttrs(ArrayAttr attrs, MLIRContext *ctx) {
  if (!attrs) {
    return {};
  }

  SmallVector<Attribute> filteredAttrs;
  bool hasRemainingAttrs = false;
  for (Attribute attr : attrs) {
    SmallVector<NamedAttribute> filteredDict;
    for (NamedAttribute namedAttr : cast<DictionaryAttr>(attr)) {
      if (!namedAttr.getName().strref().starts_with("sdy.")) {
        filteredDict.push_back(namedAttr);
      }
    }
    hasRemainingAttrs |= !filteredDict.empty();
    filteredAttrs.push_back(DictionaryAttr::get(ctx, filteredDict));
  }
  return hasRemainingAttrs ? ArrayAttr::get(ctx, filteredAttrs) : ArrayAttr{};
}

// Lazily materializes one LogicalMeshAxesOp value per logical symbol.
static TV_AxisFactor getOrCreateLogicalAxisForSymbol(
    AxisSymbol symbol, OpBuilder &axisBuilder, Location axisLoc,
    llvm::DenseMap<AxisSymbol, TV_AxisFactor> &symbolToLogicalAxis) {
  auto it = symbolToLogicalAxis.find(symbol);
  if (it != symbolToLogicalAxis.end()) {
    return it->second;
  }

  auto op = axisBuilder.create<mlir::enzyme::distributed::LogicalMeshAxesOp>(
      axisLoc, symbol.getExtent());
  auto as_factor = axis::viewAxesAsFactors(op.getAxes(), axisBuilder, axisLoc);
  symbolToLogicalAxis[symbol] = as_factor.front();
  return as_factor.front();
}

// Wraps one logical axis factor as a factor-group operand for function
// metadata.
static TV_FactorGroup getOrCreatePartitioningAxisGroup(
    AxisSymbol symbol, OpBuilder &axisBuilder, Location axisLoc,
    llvm::DenseMap<AxisSymbol, TV_AxisFactor> &symbolToLogicalAxis) {
  auto factor = getOrCreateLogicalAxisForSymbol(symbol, axisBuilder, axisLoc,
                                                symbolToLogicalAxis);
  return axis::viewFactorsAsProduct(ValueRange{factor}, axisBuilder, axisLoc);
}

// Builds per-dimension sharding indices against a stable partitioning-axis
// table.
static IndexedTensorShardingAttr buildIndexedShardingAttr(
    RankedTensorType tensorType,
    const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
        &partitioningDims,
    llvm::DenseMap<AxisSymbol, int64_t> &symbolToPartitioningAxisIdx) {
  auto *ctx = tensorType.getContext();
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

// Replaces func.func @main with distributed.function and preserves sharding
// intent.
static LogicalResult convertMainToDistributedFunction(
    ModuleOp moduleOp, func::FuncOp mainFunc, Block *&mainBlock,
    Operation *&mainScopeOp, ShardyLogicalAxisAnalysis &axisAnalysis,
    OpBuilder &axisBuilder, Location axisLoc,
    llvm::DenseMap<AxisSymbol, TV_AxisFactor> &symbolToLogicalAxis) {
  auto *ctx = moduleOp.getContext();
  auto returnOp = dyn_cast<func::ReturnOp>(mainBlock->getTerminator());
  if (!returnOp) {
    mainFunc.emitError() << "expected main terminator to be func.return";
    return failure();
  }

  llvm::DenseMap<AxisSymbol, int64_t> symbolToPartitioningAxisIdx;

  // Derive argument sharding metadata and populate the axis-index table.
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
      mainFunc.emitError() << "missing partitioning mapping for block argument "
                           << arg.getArgNumber();
      return failure();
    }

    argumentShardings.push_back(buildIndexedShardingAttr(
        tensorType, *maybePartitioning, symbolToPartitioningAxisIdx));
  }

  // Derive output sharding metadata from the yielded producer values.
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

    std::optional<TensorPartitioningAxes> maybePartitioning = std::nullopt;
    if (OpResult result = dyn_cast<OpResult>(operand.get())) {
      maybePartitioning = axisAnalysis.getTensorPartitionDims(result);
    } else if (BlockArgument arg = dyn_cast<BlockArgument>(operand.get())) {
      maybePartitioning = axisAnalysis.getTensorPartitionDims(arg);
    }
    if (!maybePartitioning) {
      mainFunc.emitError() << "missing partitioning mapping for return operand "
                           << operand.getOperandNumber();
      return failure();
    }

    outputShardings.push_back(buildIndexedShardingAttr(
        tensorType, *maybePartitioning, symbolToPartitioningAxisIdx));
  }

  // Rebuild partitioning axis SSA operands in the same index order used above.
  SmallVector<Value> orderedPartitioningAxes(
      symbolToPartitioningAxisIdx.size());
  for (const auto &[symbol, idx] : symbolToPartitioningAxisIdx) {
    orderedPartitioningAxes[idx] = getOrCreatePartitioningAxisGroup(
        symbol, axisBuilder, axisLoc, symbolToLogicalAxis);
  }

  auto argShardingsAttr =
      IndexedTensorShardingPerValueAttr::get(ctx, argumentShardings);
  auto outputShardingsAttr =
      IndexedTensorShardingPerValueAttr::get(ctx, outputShardings);

  OpBuilder builder(ctx);
  builder.setInsertionPoint(mainFunc);

  auto distributedFunction = builder.create<DistributedFunctionOp>(
      mainFunc.getLoc(), mainFunc.getSymNameAttr(),
      TypeAttr::get(mainFunc.getFunctionType()),
      ValueRange(orderedPartitioningAxes), argShardingsAttr,
      outputShardingsAttr, mainFunc.getSymVisibilityAttr(),
      dropShardyAttrs(mainFunc.getArgAttrsAttr(), ctx),
      dropShardyAttrs(mainFunc.getResAttrsAttr(), ctx));
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

struct ConvertMainToDistributedFunctionPass
    : public impl::ConvertMainToDistributedFunctionPassBase<
          ConvertMainToDistributedFunctionPass> {
  using ConvertMainToDistributedFunctionPassBase::
      ConvertMainToDistributedFunctionPassBase;

  llvm::DenseMap<AxisSymbol, TV_AxisFactor> symbolToLogicalAxis;
  std::optional<OpBuilder> axisBuilder;
  std::optional<Location> axisLoc;

  // Converts func.main once and leaves already-converted distributed.main
  // unchanged.
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
    auto mainFunc = dyn_cast<func::FuncOp>(mainScopeOp);
    if (!mainFunc) {
      return;
    }

    Block *mainBlock = mainFunctionAnalysis.getMainBlock();
    if (!mainBlock) {
      emitError(mainFunc.getLoc()) << "main function has no body";
      signalPassFailure();
      return;
    }

    axisBuilder = OpBuilder(moduleOp.getContext());
    axisBuilder->setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    axisLoc = mainFunc.getLoc();

    const auto &mainAxisAnalysis =
        getAnalysis<MainFunctionShardyLogicalAxisAnalysis>();
    if (!mainAxisAnalysis.isValid()) {
      emitError(mainFunc.getLoc())
          << "failed to build module-scoped main logical axis analysis";
      signalPassFailure();
      return;
    }

    auto axisAnalysis = mainAxisAnalysis.getAnalysis();

    if (failed(convertMainToDistributedFunction(
            moduleOp, mainFunc, mainBlock, mainScopeOp, axisAnalysis,
            *axisBuilder, *axisLoc, symbolToLogicalAxis))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
