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

// Lazily materializes one axis value per logical symbol: a LogicalMeshAxesOp,
// or a DeviceLocalAxisOp for a symbol that must never be sharded across the
// mesh. Keeping the latter in each tensor's axes, rather than omitting them,
// preserves the position of every factor within its dimension.
static TV_AxisFactor getOrCreateLogicalAxisForSymbol(
    AxisSymbol symbol, ShardyLogicalAxisAnalysis &axisAnalysis,
    OpBuilder &axisBuilder, Location axisLoc,
    llvm::DenseMap<AxisSymbol, TV_AxisFactor> &symbolToLogicalAxis) {
  auto it = symbolToLogicalAxis.find(symbol);
  if (it != symbolToLogicalAxis.end()) {
    return it->second;
  }

  Value axisValue;
  if (axisAnalysis.isUnshardable(symbol)) {
    axisValue = axisBuilder
                    .create<mlir::enzyme::distributed::DeviceLocalAxisOp>(
                        axisLoc, symbol.getExtent())
                    .getAxis();
  } else {
    axisValue = axisBuilder
                    .create<mlir::enzyme::distributed::LogicalMeshAxesOp>(
                        axisLoc, symbol.getExtent())
                    .getAxis();
  }
  auto as_factor = axis::viewAxisAsFactor(axisValue, axisBuilder, axisLoc);
  symbolToLogicalAxis[symbol] = as_factor;
  return as_factor;
}

// Wraps one logical axis factor as a factor-group operand for function
// metadata.
static TV_FactorGroup getOrCreatePartitioningAxisGroup(
    AxisSymbol symbol, ShardyLogicalAxisAnalysis &axisAnalysis,
    OpBuilder &axisBuilder, Location axisLoc,
    llvm::DenseMap<AxisSymbol, TV_AxisFactor> &symbolToLogicalAxis) {
  auto factor = getOrCreateLogicalAxisForSymbol(
      symbol, axisAnalysis, axisBuilder, axisLoc, symbolToLogicalAxis);
  return axis::viewFactorsAsProduct(factor, axisBuilder, axisLoc);
}

// The analysis' partitioning axes for a global value, if it has any.
static std::optional<TensorPartitioningAxes>
getValueDims(ShardyLogicalAxisAnalysis &axisAnalysis, Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    return axisAnalysis.getTensorPartitionDims(result);
  }
  return axisAnalysis.getTensorPartitionDims(cast<BlockArgument>(value));
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

// Replaces a func.func with a distributed.function and preserves sharding
// intent.
static LogicalResult convertFunctionToDistributedFunction(
    ModuleOp moduleOp, func::FuncOp mainFunc,
    ShardyLogicalAxisAnalysis &axisAnalysis, OpBuilder &axisBuilder,
    Location axisLoc,
    llvm::DenseMap<AxisSymbol, TV_AxisFactor> &symbolToLogicalAxis) {
  auto *ctx = moduleOp.getContext();
  Block *mainBlock = &mainFunc.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(mainBlock->getTerminator());
  if (!returnOp) {
    mainFunc.emitError() << "expected function terminator to be func.return";
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

    std::optional<TensorPartitioningAxes> maybePartitioning =
        getValueDims(axisAnalysis, operand.get());
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
        symbol, axisAnalysis, axisBuilder, axisLoc, symbolToLogicalAxis);
  }

  auto argShardingsAttr =
      IndexedTensorShardingPerValueAttr::get(ctx, argumentShardings);
  auto outputShardingsAttr =
      IndexedTensorShardingPerValueAttr::get(ctx, outputShardings);

  OpBuilder builder(ctx);
  builder.setInsertionPoint(mainFunc);

  // ShardyLogicalAxisAnalysis's reshard-identity check (buildInitialSymbols
  // in ShardyLogicalAxisAnalysis.cpp) needs a reshard operand's real
  // sdy.sharding to know which dimensions the reshard touches, and this is
  // the last point with the mesh's axis names in hand to preserve it -- once
  // conversion finishes, only our own logical-axis symbols remain. Nothing
  // else consults these attrs on this op (LowerKernels writes its own fresh
  // sdy.sharding on kernel bodies later, from our own derived metadata), so
  // passing them through unchanged doesn't create a second, driftable source
  // of truth.
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
        << "expected moved function block to end with func.return";
    return failure();
  }

  builder.setInsertionPoint(movedReturnOp);
  auto yieldOp = builder.create<DistributedYieldOp>(
      movedReturnOp.getLoc(), movedReturnOp.getOperands());
  axisAnalysis.markRewrite(movedReturnOp, yieldOp);
  movedReturnOp.erase();

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

  // What every call site of one callee shares: the callee's global argument
  // and result axes, as per-dimension factor-group values ready for casts.
  struct CalleeBinding {
    SmallVector<std::optional<TensorPartitioningAxes>> argDims, resultDims;
    SmallVector<SmallVector<Value>> argGroups, resultGroups;
  };
  llvm::DenseMap<Operation *, CalleeBinding> calleeBindings;

  // One factor-group value per tensor dimension, as a cast's axes.
  SmallVector<Value> getAxisGroups(const TensorPartitioningAxes &dims,
                                   ShardyLogicalAxisAnalysis &axisAnalysis) {
    SmallVector<Value> groups;
    for (const auto &dimSymbols : dims) {
      SmallVector<Value> factors;
      for (AxisSymbol symbol : dimSymbols) {
        factors.push_back(getOrCreateLogicalAxisForSymbol(
            symbol, axisAnalysis, *axisBuilder, *axisLoc,
            symbolToLogicalAxis));
      }
      groups.push_back(
          axis::viewFactorsAsProduct(factors, *axisBuilder, *axisLoc));
    }
    return groups;
  }

  // Computes a callee's binding once, from the analysis, for all its calls.
  FailureOr<CalleeBinding *>
  getCalleeBinding(DistributedFunctionOp callee,
                   ShardyLogicalAxisAnalysis &axisAnalysis) {
    auto [it, inserted] = calleeBindings.try_emplace(callee);
    CalleeBinding &binding = it->second;
    if (!inserted) {
      return &binding;
    }
    auto bind = [&](Value value, auto &dimsList, auto &groupsList) {
      std::optional<TensorPartitioningAxes> dims;
      SmallVector<Value> groups;
      if (isa<RankedTensorType>(value.getType())) {
        dims = getValueDims(axisAnalysis, value);
        if (!dims) {
          return false;
        }
        groups = getAxisGroups(*dims, axisAnalysis);
      }
      dimsList.push_back(std::move(dims));
      groupsList.push_back(std::move(groups));
      return true;
    };
    Block &body = callee.getBody().front();
    for (BlockArgument arg : body.getArguments()) {
      if (!bind(arg, binding.argDims, binding.argGroups)) {
        callee.emitError() << "missing partitioning mapping for argument "
                           << arg.getArgNumber();
        return failure();
      }
    }
    for (Value returned : body.getTerminator()->getOperands()) {
      if (!bind(returned, binding.resultDims, binding.resultGroups)) {
        callee.emitError() << "missing partitioning mapping for a result";
        return failure();
      }
    }
    return &binding;
  }

  // Replaces a call with a localized distributed call: local operands and
  // results bound to the callee's own axes, with casts to and from the
  // surrounding global values. The call shares the callee's sharding
  // metadata, since both index the same partitioning axes.
  LogicalResult localizeCall(func::CallOp call,
                             ShardyLogicalAxisAnalysis &axisAnalysis) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<DistributedFunctionOp>(
        call, call.getCalleeAttr());
    if (!callee) {
      return call.emitError() << "call to a non-distributed function";
    }
    auto binding = getCalleeBinding(callee, axisAnalysis);
    if (failed(binding)) {
      return failure();
    }
    OpBuilder builder(call);

    SmallVector<Value> localOperands;
    for (auto [idx, operand] : llvm::enumerate(call.getOperands())) {
      if (!(*binding)->argDims[idx]) {
        localOperands.push_back(operand);
        continue;
      }
      localOperands.push_back(
          builder
              .create<DistributedCastGlobalToLocalOp>(
                  call.getLoc(),
                  getLocalTensorType(cast<RankedTensorType>(operand.getType()),
                                     *(*binding)->argDims[idx]),
                  operand, (*binding)->argGroups[idx])
              .getOutput());
    }

    SmallVector<Type> localResultTypes;
    for (auto [idx, result] : llvm::enumerate(call.getResults())) {
      localResultTypes.push_back(
          (*binding)->resultDims[idx]
              ? getLocalTensorType(cast<RankedTensorType>(result.getType()),
                                   *(*binding)->resultDims[idx])
              : result.getType());
    }

    auto localCall = builder.create<DistributedCallOp>(
        call.getLoc(), localResultTypes, call.getCalleeAttr(), localOperands,
        callee.getPartitioningAxes(), callee.getArgumentShardingsAttr(),
        callee.getOutputShardingsAttr(), call.getArgAttrsAttr(),
        call.getResAttrsAttr());
    axisAnalysis.markRewrite(call, localCall);

    for (auto [idx, result] : llvm::enumerate(call.getResults())) {
      Value replacement = localCall.getResult(idx);
      if ((*binding)->resultDims[idx]) {
        replacement = builder
                          .create<DistributedCastLocalToGlobalOp>(
                              call.getLoc(), result.getType(), replacement,
                              (*binding)->resultGroups[idx])
                          .getOutput();
      }
      result.replaceAllUsesWith(replacement);
    }
    call.erase();
    return success();
  }

  // Localizes every call, after all functions are converted, since a call
  // verifies against a distributed function.
  LogicalResult localizeCalls(ModuleOp moduleOp,
                              ShardyLogicalAxisAnalysis &axisAnalysis) {
    SmallVector<func::CallOp> calls;
    moduleOp.walk([&](func::CallOp call) { calls.push_back(call); });
    for (func::CallOp call : calls) {
      if (failed(localizeCall(call, axisAnalysis))) {
        return failure();
      }
    }
    return success();
  }

  // Converts main and every function it transitively calls, leaving
  // already-converted distributed functions unchanged.
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

    // Snapshot the functions first: conversion erases the analyzed ops.
    SmallVector<func::FuncOp> funcs;
    for (Operation *func : axisAnalysis.getAnalyzedFunctions()) {
      funcs.push_back(cast<func::FuncOp>(func));
    }

    // Dump before any rewriting below, so the output reflects analysis of
    // Shardy IR and remains available even if a later step fails.
    if (dumpValueAxes || dumpOperationAxes) {
      for (func::FuncOp func : funcs) {
        if (dumpValueAxes) {
          distributed::dumpValueAxes(llvm::errs(), &func.getBody().front(),
                                     axisAnalysis);
        }
        if (dumpOperationAxes) {
          distributed::dumpOperationAxes(llvm::errs(), &func.getBody().front(),
                                         axisAnalysis);
        }
      }
    }

    for (func::FuncOp func : funcs) {
      if (failed(convertFunctionToDistributedFunction(
              moduleOp, func, axisAnalysis, *axisBuilder, *axisLoc,
              symbolToLogicalAxis))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(localizeCalls(moduleOp, axisAnalysis))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
