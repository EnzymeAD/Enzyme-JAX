#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "stablehlo/dialect/StablehloOps.h"

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

  for (auto [dim, partitioning] :
       llvm::enumerate(kernelOp.getPartitioningAxes())) {
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

// Shardy expects a function-shaped wrapper, but the kernel's block arguments
// are the canonical source of truth for the current local view. The kernel body
// is discarded once the rewritten function is spliced back in
// copyShardyModuleToKernelAndErase, so its ops are moved rather than cloned.
static ModuleOp kernelToModule(DistributedKernelOp kernelOp) {
  OpBuilder builder(kernelOp.getContext());
  auto shardyModule = ModuleOp::create(builder.getUnknownLoc());

  // Use the kernel body's own (possibly already partially-shrunk) block-arg
  // and yield-operand types, not the kernel op's external operand/result
  // types: those two only coincide on the first pass invocation, and diverge
  // once any factor has been lowered, which would desync operand/result
  // shapes on a second run (idempotency).
  auto &kernelBody = kernelOp.getBody().front();
  auto yield = cast<DistributedYieldOp>(kernelBody.getTerminator());
  SmallVector<Type> shardyInputTypes(kernelBody.getArgumentTypes().begin(),
                                     kernelBody.getArgumentTypes().end());
  SmallVector<Type> shardyResultTypes(yield.getOperandTypes().begin(),
                                      yield.getOperandTypes().end());

  auto fnType = FunctionType::get(kernelOp.getContext(), shardyInputTypes,
                                  shardyResultTypes);
  auto shardyFunc =
      func::FuncOp::create(builder, kernelOp.getLoc(), "kernel", fnType);

  auto &funcBody = *shardyFunc.addEntryBlock();
  for (auto [origArg, newArg] :
       llvm::zip_equal(kernelBody.getArguments(), funcBody.getArguments())) {
    // Somewhat worried about replacing uses not in the copied module,
    // but scoping rules here means every use will be overwritten anyways,
    // so going with all uses rather than region-specific uses for now.
    // mlir::replaceAllUsesInRegionWith(origArg, newArg, &funcBody);
    origArg.replaceAllUsesWith(newArg);
  }

  SmallVector<Value> results(yield.getOperands().begin(),
                             yield.getOperands().end());
  yield.erase();
  funcBody.getOperations().splice(funcBody.end(), kernelBody.getOperations());

  builder.setInsertionPointToEnd(&funcBody);
  builder.create<func::ReturnOp>(kernelOp.getLoc(), results);

  shardyModule.push_back(shardyFunc);

  return shardyModule;
}

// Shardy axis name for a kernel partitioning-axis index; index-based so no
// separate remap table is needed between the two attribute schemes.
static std::string shardyAxisName(int64_t partitioningAxisIndex) {
  return ("a" + llvm::Twine(partitioningAxisIndex)).str();
}

static llvm::SmallVector<mlir::sdy::AxisRefAttr>
shardyAxisRefsForIndices(MLIRContext *ctx, llvm::ArrayRef<int64_t> indices) {
  llvm::SmallVector<mlir::sdy::AxisRefAttr> axisRefs;
  axisRefs.reserve(indices.size());
  for (int64_t index : indices) {
    axisRefs.push_back(mlir::sdy::AxisRefAttr::get(ctx, shardyAxisName(index)));
  }
  return axisRefs;
}

// Translates one distributed.indexed_tensor_sharding into a Shardy
// TensorShardingAttr bound to `meshName`; both attributes share the same
// partitioning-axis index space, so dim/unreduced indices translate directly.
static mlir::sdy::TensorShardingAttr
translateIndexedSharding(MLIRContext *ctx, StringRef meshName,
                         IndexedTensorShardingAttr indexed) {
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
  dimShardings.reserve(indexed.getDimPartitioningAxes().size());
  for (DenseI64ArrayAttr dimAxes : indexed.getDimPartitioningAxes()) {
    dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
        ctx, shardyAxisRefsForIndices(ctx, dimAxes.asArrayRef()),
        /*is_closed=*/true));
  }

  llvm::SmallVector<mlir::sdy::AxisRefAttr> unreducedAxisRefs =
      shardyAxisRefsForIndices(ctx, indexed.getUnreducedAxes().asArrayRef());

  // TODO: assumes SUM reduction (Shardy's default); revisit if
  // IndexedTensorShardingAttr ever gains a reduction-kind field.
  return mlir::sdy::TensorShardingAttr::get(ctx, meshName, dimShardings,
                                            /*replicated_axes=*/{},
                                            unreducedAxisRefs);
}

static mlir::sdy::TensorShardingPerValueAttr
translateIndexedShardingPerValue(MLIRContext *ctx, StringRef meshName,
                                 IndexedTensorShardingPerValueAttr perValue) {
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> shardings;
  shardings.reserve(perValue.getShardings().size());
  for (IndexedTensorShardingAttr indexed : perValue.getShardings()) {
    shardings.push_back(translateIndexedSharding(ctx, meshName, indexed));
  }
  return mlir::sdy::TensorShardingPerValueAttr::get(ctx, shardings);
}

// Every partitioning-axis index appearing in any value's dim_partitioning_axes
// or unreduced_axes; used to tell which axes are "visible" on the argument or
// result side of an op.
static llvm::SmallVector<int64_t>
collectAxisIndices(IndexedTensorShardingPerValueAttr perValue) {
  if (!perValue) {
    return {};
  }
  llvm::SmallDenseSet<int64_t> seen;
  llvm::SmallVector<int64_t> indices;
  auto add = [&](int64_t index) {
    if (seen.insert(index).second) {
      indices.push_back(index);
    }
  };
  for (IndexedTensorShardingAttr indexed : perValue.getShardings()) {
    for (DenseI64ArrayAttr dimAxes : indexed.getDimPartitioningAxes()) {
      for (int64_t index : dimAxes.asArrayRef()) {
        add(index);
      }
    }
    for (int64_t index : indexed.getUnreducedAxes().asArrayRef()) {
      add(index);
    }
  }
  llvm::sort(indices);
  return indices;
}

// Partitioning axes that shard an op's operands but never reach its results
// (e.g. a matmul's contracting dimension): the op reduces these away locally,
// but Shardy's ConvertGlobalToLocal only accepts a sharded contracting dim if
// the op has an explicit sdy.all_reduce consumer over exactly those axes. See
// insertPlaceholderAllReduces, which satisfies that and is undone afterward.
static llvm::SmallVector<int64_t>
computeReductionAxes(IndexedTensorShardingPerValueAttr argumentShardings,
                     IndexedTensorShardingPerValueAttr outputShardings) {
  llvm::SmallVector<int64_t> argAxes = collectAxisIndices(argumentShardings);
  llvm::SmallDenseSet<int64_t> visible;
  for (int64_t index : collectAxisIndices(outputShardings)) {
    visible.insert(index);
  }
  llvm::SmallVector<int64_t> reductionAxes;
  for (int64_t index : argAxes) {
    if (!visible.contains(index)) {
      reductionAxes.push_back(index);
    }
  }
  return reductionAxes;
}

// Adds `extraAxisIndices` to `sharding`'s unreduced axes (no-op if empty).
static mlir::sdy::TensorShardingAttr
withAdditionalUnreducedAxes(MLIRContext *ctx,
                            mlir::sdy::TensorShardingAttr sharding,
                            llvm::ArrayRef<int64_t> extraAxisIndices) {
  if (extraAxisIndices.empty()) {
    return sharding;
  }
  llvm::SmallVector<mlir::sdy::AxisRefAttr> unreduced(
      sharding.getUnreducedAxes().begin(), sharding.getUnreducedAxes().end());
  llvm::append_range(unreduced,
                     shardyAxisRefsForIndices(ctx, extraAxisIndices));
  return sharding.replaceUnreducedAxes(unreduced);
}

// Marks `op`'s own result sharding as unreduced along `reductionAxes` (so its
// operand-side sharded contracting dims are accepted by ConvertGlobalToLocal),
// then inserts a placeholder sdy.all_reduce consumer over those axes -- the
// sole structural requirement Shardy imposes on a sharded-contracting op.
// This all_reduce is never meant to survive: it gets materialized into a real
// collective by ConvertGlobalToLocal, which stripPlaceholderAllReduces then
// removes, since the actual cross-device reduction is handled outside the
// kernel (by a distributed.Collective), not by Shardy.
static void insertPlaceholderAllReduces(
    MLIRContext *ctx, StringRef meshName, Operation *op,
    mlir::sdy::TensorShardingPerValueAttr baseResultShardings,
    llvm::ArrayRef<int64_t> reductionAxes) {
  if (reductionAxes.empty()) {
    op->setAttr("sdy.sharding", baseResultShardings);
    return;
  }

  llvm::SmallVector<mlir::sdy::TensorShardingAttr> opShardings;
  opShardings.reserve(baseResultShardings.getShardings().size());
  for (auto sharding : baseResultShardings.getShardings()) {
    opShardings.push_back(
        withAdditionalUnreducedAxes(ctx, sharding, reductionAxes));
  }
  op->setAttr("sdy.sharding",
              mlir::sdy::TensorShardingPerValueAttr::get(ctx, opShardings));

  auto reductionAxisRefs = shardyAxisRefsForIndices(ctx, reductionAxes);
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (auto [result, outSharding] :
       llvm::zip_equal(op->getResults(), baseResultShardings.getShardings())) {
    // TODO: assumes SUM reduction (Shardy's default); revisit if
    // IndexedTensorShardingAttr ever gains a reduction-kind field.
    auto allReduce = builder.create<mlir::sdy::AllReduceOp>(
        op->getLoc(), result.getType(), result, reductionAxisRefs,
        mlir::sdy::ReductionOp::SUM, outSharding);
    result.replaceAllUsesExcept(allReduce.getResult(), allReduce);
  }
}

// Placeholder all_reduce ops (see insertPlaceholderAllReduces) get lowered by
// ConvertGlobalToLocal into real stablehlo.all_reduce collectives; strip
// those back out since a distributed kernel body must never itself contain a
// collective -- the actual cross-device reduction happens outside the kernel.
static void stripPlaceholderAllReduces(ModuleOp shardyModule) {
  llvm::SmallVector<stablehlo::AllReduceOp> allReduces;
  shardyModule.walk([&](stablehlo::AllReduceOp allReduce) {
    allReduces.push_back(allReduce);
  });
  for (auto allReduce : allReduces) {
    allReduce.getResult(0).replaceAllUsesWith(allReduce.getOperand(0));
    allReduce.erase();
  }
}

/**
 * Constructs the Shardy sharding attributes for the module based on the
 * sharding factors per dimension.
 *
 * Note: implemented along the logic that a sharding dimension is either
 * embarassingly parallel or an associative reduction, so even if the paralleism
 * isn't contiguous the rewrite is the same (with different operand tensors /
 * blocks passed in from the collectives). This may not hold up to all sharding
 * dimensions, or may simply not be correct.
 */
static void constructShardyAttributes(DistributedKernelOp originalKernel,
                                      ModuleOp shardyModule,
                                      llvm::SmallVector<int> &shardingFactors) {
  MLIRContext *ctx = shardyModule.getContext();
  constexpr llvm::StringLiteral kMeshName = "mesh";

  // Add a sdy.mesh op to the module with axes "a0", "a1", ... corresponding to
  // the sharding factors (and sized accordingly). Future attributes will
  // reference this mesh to translate indices directly into mesh axis names.
  llvm::SmallVector<mlir::sdy::MeshAxisAttr> meshAxes;
  meshAxes.reserve(shardingFactors.size());
  for (auto [index, factor] : llvm::enumerate(shardingFactors)) {
    meshAxes.push_back(
        mlir::sdy::MeshAxisAttr::get(ctx, shardyAxisName(index), factor));
  }
  OpBuilder meshBuilder(ctx);
  meshBuilder.setInsertionPointToStart(shardyModule.getBody());
  meshBuilder.create<mlir::sdy::MeshOp>(
      shardyModule.getLoc(), kMeshName,
      mlir::sdy::MeshAttr::get(ctx, meshAxes));

  auto shardyFunc = shardyModule.lookupSymbol<func::FuncOp>("kernel");
  if (!shardyFunc) {
    shardyModule.emitError() << "missing shardy kernel wrapper function";
    return;
  }

  // Need to add attributes for the func arg, return shardings based on the
  // original kernel
  mlir::sdy::TensorShardingPerValueAttr argShardings =
      translateIndexedShardingPerValue(ctx, kMeshName,
                                       originalKernel.getArgumentShardings());
  for (auto [argIndex, blockArg] : llvm::enumerate(shardyFunc.getArguments())) {
    mlir::sdy::setSharding(blockArg, argShardings.getSharding(argIndex));
  }

  mlir::sdy::TensorShardingPerValueAttr outputShardings =
      translateIndexedShardingPerValue(ctx, kMeshName,
                                       originalKernel.getOutputShardings());
  mlir::sdy::setFuncResultShardings(shardyFunc, outputShardings);

  // Every op in the body is annotated explicitly (no reliance on producer/
  // consumer sharding matching): output_shardings pins the op's own results
  // (with a placeholder all_reduce for any reduction axes -- see
  // insertPlaceholderAllReduces), argument_shardings pins each operand
  // independently via a per-use sharding_constraint, since Shardy has no
  // per-op operand-sharding concept.
  //
  // Collect the ops to process before mutating: insertPlaceholderAllReduces
  // inserts a new sdy.all_reduce op right after its target, and walking a
  // region while inserting into it is asking for trouble (the walk may end
  // up visiting the newly-inserted ops too), so mutation is done in a
  // separate pass over a fixed snapshot of operations instead.
  llvm::SmallVector<Operation *> opsToProcess;
  shardyModule.getRegion().walk([&](Operation *op) {
    if (op->hasAttr("distributed.argument_shardings") ||
        op->hasAttr("distributed.output_shardings")) {
      opsToProcess.push_back(op);
    }
  });

  for (Operation *op : opsToProcess) {
    auto argumentShardings =
        op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
            "distributed.argument_shardings");

    if (auto outputShardings =
            op->getAttrOfType<IndexedTensorShardingPerValueAttr>(
                "distributed.output_shardings")) {
      auto baseResultShardings =
          translateIndexedShardingPerValue(ctx, kMeshName, outputShardings);
      auto reductionAxes =
          computeReductionAxes(argumentShardings, outputShardings);
      insertPlaceholderAllReduces(ctx, kMeshName, op, baseResultShardings,
                                  reductionAxes);
    }

    if (!argumentShardings) {
      continue;
    }
    OpBuilder opBuilder(op);
    for (auto [operandIndex, indexed] :
         llvm::enumerate(argumentShardings.getShardings())) {
      Value operand = op->getOperand(operandIndex);
      auto sharding = translateIndexedSharding(ctx, kMeshName, indexed);
      // ConvertGlobalToLocal rejects any surviving sharding_constraint op, and
      // ApplyShardingConstraintsPass only folds one away when its input has
      // no sharding of its own yet -- which is never true here, since the
      // operand's producer (or block arg) always already carries one. So we
      // only insert the constraint when it actually differs from what the
      // operand already has, keeping the common (already-consistent) case a
      // no-op.
      if (sharding == mlir::sdy::getSharding(operand)) {
        continue;
      }
      auto constraint = opBuilder.create<mlir::sdy::ShardingConstraintOp>(
          op->getLoc(), operand, sharding);
      op->setOperand(operandIndex, constraint.getResult());
    }
  }
}

// The bookkeeping step is intentionally small: once Shardy rewrites the
// temporary function, we only need to re-normalize the kernel's block argument
// tensor shapes for the axes whose shardability has just been lowered from the
// in-kernel partitioning basis.
static void updateKernelArgumentTypes(DistributedKernelOp kernelOp,
                                      FactorsPerDim &shardableFactors) {
  auto &entryBlock = kernelOp.getBody().front();
  auto argumentShardings = kernelOp.getArgumentShardings().getShardings();
  for (auto [argIndex, arg] : llvm::enumerate(entryBlock.getArguments())) {
    auto rankedType = dyn_cast<RankedTensorType>(arg.getType());
    if (!rankedType) {
      continue;
    }

    // A tensor dim's partitioning-axis indices (from argument_shardings) are
    // NOT the same as the dim's own index -- e.g. a dot_general operand's
    // contracting dim can be partitioning-axis 3 while its own tensor dim is
    // 0 -- so shardableFactors (keyed by partitioning-axis index) must be
    // looked up per dim via this mapping, not by the dim index directly.
    SmallVector<int64_t> updatedShape(rankedType.getShape());
    for (auto [dim, dimAxes] : llvm::enumerate(
             argumentShardings[argIndex].getDimPartitioningAxes())) {
      if (dim >= static_cast<int64_t>(updatedShape.size())) {
        break;
      }

      int64_t factorExtent = 1;
      for (int64_t axisIndex : dimAxes.asArrayRef()) {
        if (axisIndex < 0 ||
            axisIndex >= static_cast<int64_t>(shardableFactors.size())) {
          continue;
        }
        for (auto factor : shardableFactors[axisIndex]) {
          factorExtent *= static_cast<int64_t>(axis::getFactorExtent(factor));
        }
      }
      if (factorExtent > 1 && updatedShape[dim] % factorExtent == 0) {
        updatedShape[dim] /= factorExtent;
      }
    }

    if (updatedShape != rankedType.getShape()) {
      arg.setType(
          RankedTensorType::get(updatedShape, rankedType.getElementType()));
    }
  }
}

// Once factors have been sharded over, they no longer belong in the kernel's
// per-dim axis product; a sharding axis fully sharded over becomes an empty
// product (extent 1) rather than being dropped, since removing an operand
// entirely would desync the axis indices referenced by the sharding attrs.
// Dims with nothing removed this run keep their existing operand unchanged,
// so re-running the pass does not create redundant axis.product ops. Inserted
// at module scope (like the axis ops these reference/replace already are),
// not local to the kernel, so a follow-up `--cse --canonicalize
// --stabilize-axis-order` can merge the many identical empty products that
// naturally arise across sibling kernels.
static void
removeShardedFactorsFromPartitioningAxes(DistributedKernelOp kernelOp,
                                         FactorsPerDim &shardedFactors,
                                         FactorsPerDim &remainingFactors) {
  OpBuilder builder(kernelOp.getContext());
  builder.setInsertionPointToStart(
      kernelOp->getParentOfType<ModuleOp>().getBody());

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
    results.assign(returnOp.getOperands().begin(),
                   returnOp.getOperands().end());
    returnOp.erase();
  }
  oldBody.getOperations().splice(oldBody.end(), funcBody.getOperations());

  auto yield =
      OpBuilder(kernelOp.getContext())
          .create<DistributedYieldOp>(kernelOp.getLoc(), TypeRange{}, results);
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
static void runShardyLowering(ModuleOp shardyModule, bool verify) {
  PassManager pm(shardyModule.getContext());
  // With maximal logical parallelism (one device id per tensor element),
  // MLIR's verifier can spend a very long time walking the resulting
  // hundred-thousand-plus-element replica-group attributes; skip it here by
  // default (verify defaults to false at the pass level) and only pay that
  // cost when explicitly asked for.
  pm.enableVerifier(verify);
  // Our sharding_constraint ops always pin fully-closed shardings that match
  // the kernel's own local ABI, so this should always fold them away rather
  // than emit a reshard/collective (ConvertGlobalToLocal cannot handle a
  // surviving sharding_constraint op on its own).
  pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());
  mlir::sdy::ConvertGlobalToLocalPassOptions convertOptions;
  // Emit collectives with mesh-axes-based (symbolic) replica groups instead
  // of enumerating every device id literally -- the literal form is what
  // makes maximal logical parallelism's huge axis extents so slow to build
  // and verify.
  convertOptions.enableRGV3 = true;
  pm.addPass(mlir::sdy::createConvertGlobalToLocalPass(convertOptions));
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
    llvm::SmallVector<int> parallelismPerDim;
    for (const auto &factors : shardableParts) {
      int dimParallelism = 1;
      for (auto factor : factors) {
        dimParallelism *= axis::getFactorExtent(factor);
      }
      parallelismPerDim.push_back(dimParallelism);
    }

    // The temporary func.func is intentionally a thin wrapper: Shardy works on
    // function-shaped IR, while the kernel op itself carries the local ABI
    // types.
    auto shardyModule = kernelToModule(kernelOp);
    constructShardyAttributes(kernelOp, shardyModule, parallelismPerDim);
    // debug logging option
    if (dumpShardyModules) {
      llvm::dbgs() << "Dumping Shardy module:\n";
      llvm::dbgs() << shardyModule << "\n";
    }
    // Transform and copy back the module
    runShardyLowering(shardyModule, verifyShardyLowering);
    stripPlaceholderAllReduces(shardyModule);
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
