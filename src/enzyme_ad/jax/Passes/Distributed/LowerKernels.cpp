#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_LOWERKERNELSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {
using FactorsPerDim = llvm::SmallVector<
    llvm::SmallVector<mlir::TypedValue<axis::AxisFactorType>>>;

// A kernel's external interface is already in the local / maximally-sharded
// form. The temporary func.func we build for Shardy should therefore mirror the
// kernel ABI instead of inventing a second, synthetic ABI. Shardy itself is
// global-shape oriented, but it will work off the *same* tensor metadata we
// already carry on the kernel block arguments and yield values.
static bool isShardableFactor(mlir::TypedValue<axis::AxisFactorType> factor,
                              bool lowerLogical) {
  auto provenance = axis::getFactorProvenanceAxis(factor);
  if (failed(provenance)) {
    return false;
  }

  // Only logical mesh axes are optionally shardable. Replication and
  // device-local axes never participate in sharding lowering.
  if (isa<LogicalMeshAxisType>((*provenance).getType())) {
    return lowerLogical;
  }
  return false;
}

static void splitPartitioningAxesByShardability(
    DistributedKernelOp kernelOp, bool lowerLogical,
    FactorsPerDim &shardableParts, FactorsPerDim &nonShardableParts) {
  shardableParts.clear();
  nonShardableParts.clear();

  for (auto [dim, partitioning] : llvm::enumerate(kernelOp.getPartitioningAxes())) {
    (void)dim;
    auto &shardable = shardableParts.emplace_back();
    auto &nonShardable = nonShardableParts.emplace_back();

    auto factorGroup =
        mlir::cast<mlir::TypedValue<axis::FactorGroupType>>(partitioning);
    auto factors = axis::getProductProvenanceFactors(factorGroup);
    if (failed(factors)) {
      continue;
    }

    for (mlir::TypedValue<axis::AxisFactorType> factor : *factors) {
      if (isShardableFactor(factor, lowerLogical)) {
        shardable.push_back(factor);
      } else {
        nonShardable.push_back(factor);
      }
    }
  }
}

// Shardy expects a function-shaped wrapper, but the kernel's block arguments are
// the canonical source of truth for the current local view. The kernel body is
// discarded once the rewritten function is spliced back in
// copyShardyModuleToKernelAndErase, so its ops are moved rather than cloned.
static ModuleOp kernelToShardyModule(DistributedKernelOp kernelOp) {
  OpBuilder builder(kernelOp.getContext());
  auto shardyModule = ModuleOp::create(builder.getUnknownLoc());

  // Note: use the named $arguments accessor, not getOperandTypes(), since
  // AttrSizedOperandSegments packs partitioning_axes into the same operand
  // range.
  SmallVector<Type> shardyInputTypes(kernelOp.getArguments().getTypes().begin(),
                                    kernelOp.getArguments().getTypes().end());
  SmallVector<Type> shardyResultTypes(kernelOp.getResultTypes().begin(),
                                     kernelOp.getResultTypes().end());

  auto fnType = FunctionType::get(kernelOp.getContext(), shardyInputTypes,
                                  shardyResultTypes);
  auto shardyFunc = func::FuncOp::create(builder, kernelOp.getLoc(), "kernel",
                                        fnType);

  auto &kernelBody = kernelOp.getBody().front();
  auto &funcBody = *shardyFunc.addEntryBlock();
  for (auto [origArg, newArg] :
       llvm::zip_equal(kernelBody.getArguments(), funcBody.getArguments())) {
    origArg.replaceAllUsesWith(newArg);
  }

  SmallVector<Value> results;
  if (auto yield = dyn_cast<DistributedYieldOp>(kernelBody.getTerminator())) {
    results.assign(yield.getOperands().begin(), yield.getOperands().end());
    yield.erase();
  }
  funcBody.getOperations().splice(funcBody.end(), kernelBody.getOperations());

  builder.setInsertionPointToEnd(&funcBody);
  builder.create<func::ReturnOp>(kernelOp.getLoc(), results);

  shardyModule.push_back(shardyFunc);
  return shardyModule;
}

// The bookkeeping step is intentionally small: once Shardy rewrites the
// temporary function, we only need to re-normalize the kernel's block argument
// tensor shapes for the axes whose shardability has just been lowered from the
// in-kernel partitioning basis.
static void updateKernelArgumentTypes(DistributedKernelOp kernelOp,
                                     FactorsPerDim &shardableFactors) {
  auto &entryBlock = kernelOp.getBody().front();
  for (auto [argIndex, arg] : llvm::enumerate(entryBlock.getArguments())) {
    auto rankedType = dyn_cast<RankedTensorType>(arg.getType());
    if (!rankedType) {
      continue;
    }

    SmallVector<int64_t> updatedShape(rankedType.getShape());
    for (auto [dim, factors] : llvm::enumerate(shardableFactors)) {
      if (dim >= static_cast<int64_t>(updatedShape.size())) {
        break;
      }

      int64_t factorExtent = 1;
      for (auto factor : factors) {
        factorExtent *= static_cast<int64_t>(axis::getFactorExtent(factor));
      }
      if (factorExtent > 1 && updatedShape[dim] % factorExtent == 0) {
        updatedShape[dim] /= factorExtent;
      }
    }

    if (updatedShape != rankedType.getShape()) {
      arg.setType(RankedTensorType::get(updatedShape,
                                        rankedType.getElementType()));
    }

    (void)argIndex;
  }
}

// Once factors have been sharded over, they no longer belong in the kernel's
// per-dim axis product; a sharding axis fully sharded over becomes an empty
// product (extent 1) rather than being dropped, since removing an operand
// entirely would desync the axis indices referenced by the sharding attrs.
// Dims with nothing removed this run keep their existing operand unchanged,
// so re-running the pass does not create redundant axis.product ops.
static void removeShardedFactorsFromPartitioningAxes(
    DistributedKernelOp kernelOp, FactorsPerDim &shardedFactors,
    FactorsPerDim &remainingFactors) {
  OpBuilder builder(kernelOp.getContext());
  builder.setInsertionPoint(kernelOp);

  ValueRange existingPartitioningAxes = kernelOp.getPartitioningAxes();
  SmallVector<Value> newPartitioningAxes;
  newPartitioningAxes.reserve(remainingFactors.size());
  for (auto [dim, factors] : llvm::enumerate(remainingFactors)) {
    if (shardedFactors[dim].empty()) {
      newPartitioningAxes.push_back(existingPartitioningAxes[dim]);
      continue;
    }
    newPartitioningAxes.push_back(
        axis::viewFactorsAsProduct(factors, builder, kernelOp.getLoc()));
  }
  kernelOp.getPartitioningAxesMutable().assign(newPartitioningAxes);
}

// We materialize a plain func.func wrapper only so we can invoke Shardy's
// canonical lowering pipeline. After that pass runs, we splice the rewritten
// body back into the kernel's block and keep the kernel's own local metadata as
// the source of truth.
static void copyShardyModuleToKernelAndErase(ModuleOp shardyModule,
                                            DistributedKernelOp kernelOp) {
  auto shardyFunc = shardyModule.lookupSymbol<func::FuncOp>("kernel");
  if (!shardyFunc) {
    shardyModule.emitError() << "missing shardy kernel wrapper function";
    return;
  }

  auto &funcBody = shardyFunc.getBody().front();
  auto &oldBody = kernelOp.getBody().front();
  oldBody.clear();

  for (auto [funcArg, kernelArg] :
       llvm::zip_equal(funcBody.getArguments(), oldBody.getArguments())) {
    funcArg.replaceAllUsesWith(kernelArg);
  }

  SmallVector<Value> results;
  if (auto returnOp = dyn_cast<func::ReturnOp>(funcBody.getTerminator())) {
    results.assign(returnOp.getOperands().begin(), returnOp.getOperands().end());
    returnOp.erase();
  }
  oldBody.getOperations().splice(oldBody.end(), funcBody.getOperations());

  auto yield = OpBuilder(kernelOp.getContext())
                  .create<DistributedYieldOp>(kernelOp.getLoc(), TypeRange{},
                                             results);
  oldBody.push_back(yield);

  shardyModule.erase();
}

/**
 * When sharding over an axis, we expect to see groups in the form
 * <sharding_axes> * serialize_axis, where serialize axis is the contiguousmost.
 * But this still leaves some problems: we can have multiple axes on the same
 * tensor dimension (induced from i.e. a reshape), leading to a composite
 * sharding something along the lines of <sharding_axes> * serialize_ax *
 * <sharding_axes> * serialize_ax. This doesn't play very nice with Shardy
 * rewrite capabilities, which just takes a number of dims to shard over.
 *
 * Treatement of these cases depends on the semantics of the op / axis.
 * If the axis is embarassingly parallel (pass through), we don't care: we can
 * pretend the serialize axes can be shifted to the end.
 * If the axis is an associative reduction, we can still shift the serialize
 * axes to the end, but this results in a reassociation of the reduction.
 * If the axis does not parallelize, we shouldn't be sharding over it in the
 * first place.
 *
 * So as far as the computation rewriting goes, we can just take the total
 * parallelism over the axis as a shard count. However, we will have to be
 * very careful about communication and slicing ops.
 */

/**
 * Invokes Shardy's ConvertGlobalToLocal pass and then drop the mesh/sharding
 * metadata from the synthetic wrapper.
 */
static void runShardyLowering(ModuleOp shardyModule) {
  PassManager pm(shardyModule.getContext());
  pm.addPass(mlir::sdy::createConvertGlobalToLocalPass());
  pm.addPass(mlir::sdy::createDropShardingAndMeshPass());
  if (failed(pm.run(shardyModule))) {
    shardyModule.emitError() << "Shardy lowering failed";
  }
}

struct LowerKernelsPass : public impl::LowerKernelsPassBase<LowerKernelsPass> {
  using LowerKernelsPassBase::LowerKernelsPassBase;

  // Shards a single kernel by splitting its partitioning factors into the
  // subset that is safe to lower now and the subset that should remain on the
  // local kernel view. The sharding decision is based on axis kind: logical
  // mesh axes are optionally shardable, while replication and device-local axes
  // are never lowered through Shardy.
  void lowerKernel(DistributedKernelOp kernelOp, Operation *mainScopeOp,
                   bool lowerLogical) {
    FactorsPerDim shardableParts;
    FactorsPerDim nonShardableParts;
    splitPartitioningAxesByShardability(kernelOp, lowerLogical, shardableParts,
                                       nonShardableParts);

    // The temporary func.func is intentionally a thin wrapper: Shardy works on
    // function-shaped IR, while the kernel op itself carries the local ABI types.
    auto shardyModule = kernelToShardyModule(kernelOp);
    runShardyLowering(shardyModule);
    copyShardyModuleToKernelAndErase(shardyModule, kernelOp);
    updateKernelArgumentTypes(kernelOp, shardableParts);
    removeShardedFactorsFromPartitioningAxes(kernelOp, shardableParts,
                                             nonShardableParts);

    (void)mainScopeOp;
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

    Operation *mainScopeOp = mainFunctionAnalysis.getMainFunctionOp();
    if (!mainScopeOp) {
      emitWarning(module_op.getLoc())
          << "main is not a function-like op; skipping pass";
      return;
    }

    module_op.walk([&](DistributedKernelOp kernel_op) {
      lowerKernel(kernel_op, mainScopeOp, lowerLogicalAxes);
    });
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
