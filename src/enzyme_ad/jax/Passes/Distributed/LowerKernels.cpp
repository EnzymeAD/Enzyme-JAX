#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
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
  assert(succeeded(provenance) &&
         "Expected factor to have a valid provenance axis");

  // Only logical mesh axes are optionally shardable. Replication and
  // device-local axes never participate in sharding lowering.
  if (isa<LogicalMeshAxisType>((*provenance).getType())) {
    return lowerLogical;
  }

  return !isa<DeviceLocalAxisType, ReplicationAxisType>(
      (*provenance).getType());
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

// The Shardy mesh a kernel is lowered against, plus how each of the kernel's
// partitioning-axis slots maps onto that mesh's axes.
//
// When every shardable factor comes from one physical mesh, the Shardy mesh is
// that physical mesh: each physical axis is split into "atoms" (Shardy axes)
// so that every factor is a whole number of atoms, and the device id Shardy
// sees (stablehlo.partition_id) is the real, row-major device id. Axes no
// factor uses stay in the mesh as replication, which Shardy leaves implicit.
// A slot then maps to the atoms of its shardable factors, major-first, which
// is also the order Shardy computes a dimension's tile index in.
//
// Otherwise (logical axes, which have no device numbering yet) each slot is a
// single synthetic axis sized by its shardable parallelism.
struct KernelMesh {
  llvm::SmallVector<mlir::sdy::MeshAxisAttr> axes;
  llvm::SmallVector<llvm::SmallVector<mlir::sdy::AxisRefAttr>> slotAxes;

  bool isShardable(int64_t slot) const { return !slotAxes[slot].empty(); }
};

static KernelMesh buildSyntheticKernelMesh(MLIRContext *ctx,
                                           const FactorsPerDim &shardableParts) {
  KernelMesh mesh;
  for (auto [slot, factors] : llvm::enumerate(shardableParts)) {
    int64_t parallelism = 1;
    for (auto factor : factors) {
      parallelism *= axis::getFactorExtent(factor);
    }
    std::string name = ("a" + llvm::Twine(slot)).str();
    mesh.axes.push_back(mlir::sdy::MeshAxisAttr::get(ctx, name, parallelism));
    mesh.slotAxes.emplace_back();
    if (!factors.empty()) {
      mesh.slotAxes.back().push_back(mlir::sdy::AxisRefAttr::get(ctx, name));
    }
  }
  return mesh;
}

// Returns std::nullopt if the shardable factors aren't all physical factors of
// one mesh whose device ids are row-major over its axes (the numbering Shardy
// assumes), or if two factors overlap on one physical axis.
static std::optional<KernelMesh>
buildPhysicalKernelMesh(DistributedKernelOp kernelOp,
                        const FactorsPerDim &shardableParts) {
  MLIRContext *ctx = kernelOp.getContext();
  PhysicalMeshOp meshOp;
  // Physical axis index -> the shardable factors on it.
  llvm::DenseMap<int64_t, llvm::SmallVector<mlir::TypedValue<axis::AxisFactorType>>>
      factorsByAxis;
  for (const auto &factors : shardableParts) {
    for (auto factor : factors) {
      if (axis::getFactorExtent(factor) == 1) {
        continue;
      }
      auto provenance = axis::getFactorProvenanceAxis(factor);
      auto axisResult = dyn_cast<OpResult>(Value(*provenance));
      auto getAxes =
          axisResult ? dyn_cast<GetPhysicalMeshAxesOp>(axisResult.getOwner())
                     : GetPhysicalMeshAxesOp();
      if (!getAxes) {
        return std::nullopt;
      }
      auto factorMesh = SymbolTable::lookupNearestSymbolFrom<PhysicalMeshOp>(
          getAxes, getAxes.getPhysicalMeshAttr());
      if (!factorMesh || (meshOp && factorMesh != meshOp)) {
        return std::nullopt;
      }
      meshOp = factorMesh;
      factorsByAxis[axisResult.getResultNumber()].push_back(factor);
    }
  }
  if (!meshOp) {
    return std::nullopt;
  }

  llvm::SmallVector<PhysicalCommAxisType> physicalAxes;
  for (Attribute axisAttr : meshOp.getAxesAttr()) {
    physicalAxes.push_back(
        cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));
  }
  int64_t suffixExtent = 1;
  for (auto axisType : llvm::reverse(physicalAxes)) {
    if (axisType.getIdStride() != suffixExtent) {
      return std::nullopt;
    }
    suffixExtent *= axisType.getExtent();
  }

  KernelMesh mesh;
  mesh.slotAxes.resize(shardableParts.size());
  llvm::DenseMap<Value, mlir::sdy::AxisRefAttr> atomOfFactor;
  for (auto [axisIndex, axisType] : llvm::enumerate(physicalAxes)) {
    auto factors = factorsByAxis.lookup(axisIndex);
    llvm::sort(factors, [](auto lhs, auto rhs) {
      return axis::getFactorStride(lhs) > axis::getFactorStride(rhs);
    });
    // Walk the axis major to minor, emitting an atom for each factor and for
    // each gap between factors (digits no factor uses).
    int64_t remaining = axisType.getExtent();
    int64_t atomCount = 0;
    auto addAtom = [&](int64_t extent) {
      std::string name =
          ("p" + llvm::Twine(axisIndex) + "_" + llvm::Twine(atomCount++)).str();
      mesh.axes.push_back(mlir::sdy::MeshAxisAttr::get(ctx, name, extent));
      return mlir::sdy::AxisRefAttr::get(ctx, name);
    };
    for (auto factor : factors) {
      int64_t extent = axis::getFactorExtent(factor);
      int64_t stride = axis::getFactorStride(factor);
      if (remaining % (extent * stride) != 0) {
        return std::nullopt;
      }
      if (remaining != extent * stride) {
        addAtom(remaining / (extent * stride));
      }
      atomOfFactor[factor] = addAtom(extent);
      remaining = stride;
    }
    if (remaining != 1) {
      addAtom(remaining);
    }
  }

  for (auto [slot, factors] : llvm::enumerate(shardableParts)) {
    for (auto factor : factors) {
      auto atom = atomOfFactor.find(factor);
      if (atom != atomOfFactor.end()) {
        mesh.slotAxes[slot].push_back(atom->second);
      }
    }
  }
  return mesh;
}

static KernelMesh buildKernelMesh(DistributedKernelOp kernelOp,
                                  const FactorsPerDim &shardableParts) {
  if (auto physical = buildPhysicalKernelMesh(kernelOp, shardableParts)) {
    return *physical;
  }
  return buildSyntheticKernelMesh(kernelOp.getContext(), shardableParts);
}

// Only a slot whose factors are genuinely shardable (see isShardableFactor)
// ever becomes Shardy-visible axis-refs: a slot that's purely
// DeviceLocal/Replication never needs cross-device identity, so it maps to no
// axes at all (rather than being named like any other slot). This keeps two
// independently-numbered local slots for what's really the same
// permutation/need-replication factor (see ShardyLogicalAxisAnalysis's own
// comment on why it never unifies them) from ever looking like a genuine,
// unreconciled sharding conflict to Shardy's own legality checks.
static llvm::SmallVector<mlir::sdy::AxisRefAttr>
shardyAxisRefsForIndices(llvm::ArrayRef<int64_t> indices,
                         const KernelMesh &mesh) {
  llvm::SmallVector<mlir::sdy::AxisRefAttr> axisRefs;
  for (int64_t index : indices) {
    llvm::append_range(axisRefs, mesh.slotAxes[index]);
  }
  return axisRefs;
}

// Translates one distributed.indexed_tensor_sharding into a Shardy
// TensorShardingAttr bound to `meshName`; both attributes share the same
// partitioning-axis index space, so dim/unreduced indices translate directly.
static mlir::sdy::TensorShardingAttr
translateIndexedSharding(MLIRContext *ctx, StringRef meshName,
                         IndexedTensorShardingAttr indexed,
                         const KernelMesh &mesh) {
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
  dimShardings.reserve(indexed.getDimPartitioningAxes().size());
  for (DenseI64ArrayAttr dimAxes : indexed.getDimPartitioningAxes()) {
    dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
        ctx,
        shardyAxisRefsForIndices(dimAxes.asArrayRef(), mesh),
        /*is_closed=*/true));
  }

  llvm::SmallVector<mlir::sdy::AxisRefAttr> unreducedAxisRefs =
      shardyAxisRefsForIndices(indexed.getUnreducedAxes().asArrayRef(), mesh);

  // TODO: assumes SUM reduction (Shardy's default); revisit if
  // IndexedTensorShardingAttr ever gains a reduction-kind field.
  return mlir::sdy::TensorShardingAttr::get(ctx, meshName, dimShardings,
                                            /*replicated_axes=*/{},
                                            unreducedAxisRefs);
}

static mlir::sdy::TensorShardingPerValueAttr
translateIndexedShardingPerValue(MLIRContext *ctx, StringRef meshName,
                                 IndexedTensorShardingPerValueAttr perValue,
                                 const KernelMesh &mesh) {
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> shardings;
  shardings.reserve(perValue.getShardings().size());
  for (IndexedTensorShardingAttr indexed : perValue.getShardings()) {
    shardings.push_back(
        translateIndexedSharding(ctx, meshName, indexed, mesh));
  }
  return mlir::sdy::TensorShardingPerValueAttr::get(ctx, shardings);
}

// Every genuinely shardable partitioning-axis index appearing in any value's
// dim_partitioning_axes or unreduced_axes; used to tell which axes are
// "visible" on the argument or result side of an op. Non-shardable indices
// are excluded for the same reason shardyAxisRefsForIndices drops them: they
// carry no cross-device identity worth comparing.
static llvm::SmallVector<int64_t>
collectAxisIndices(IndexedTensorShardingPerValueAttr perValue,
                   const KernelMesh &mesh) {
  if (!perValue) {
    return {};
  }
  llvm::SmallDenseSet<int64_t> seen;
  llvm::SmallVector<int64_t> indices;
  auto add = [&](int64_t index) {
    if (mesh.isShardable(index) && seen.insert(index).second) {
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
//
// Only shardable axes are considered: a non-shardable slot "vanishing"
// between the argument and output sides (e.g. a permutation factor's
// independently-numbered operand/result local symbols, see
// ShardyLogicalAxisAnalysis) never actually reduced anything and needs no
// placeholder all-reduce.
static llvm::SmallVector<int64_t>
computeReductionAxes(IndexedTensorShardingPerValueAttr argumentShardings,
                     IndexedTensorShardingPerValueAttr outputShardings,
                     const KernelMesh &mesh) {
  llvm::SmallVector<int64_t> argAxes =
      collectAxisIndices(argumentShardings, mesh);
  llvm::SmallDenseSet<int64_t> visible;
  for (int64_t index : collectAxisIndices(outputShardings, mesh)) {
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
                            llvm::ArrayRef<int64_t> extraAxisIndices,
                            const KernelMesh &mesh) {
  if (extraAxisIndices.empty()) {
    return sharding;
  }
  llvm::SmallVector<mlir::sdy::AxisRefAttr> unreduced(
      sharding.getUnreducedAxes().begin(), sharding.getUnreducedAxes().end());
  llvm::append_range(unreduced,
                     shardyAxisRefsForIndices(extraAxisIndices, mesh));
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
    llvm::ArrayRef<int64_t> reductionAxes,
    const KernelMesh &mesh) {
  if (reductionAxes.empty()) {
    op->setAttr("sdy.sharding", baseResultShardings);
    return;
  }

  llvm::SmallVector<mlir::sdy::TensorShardingAttr> opShardings;
  opShardings.reserve(baseResultShardings.getShardings().size());
  for (auto sharding : baseResultShardings.getShardings()) {
    opShardings.push_back(withAdditionalUnreducedAxes(
        ctx, sharding, reductionAxes, mesh));
  }
  op->setAttr("sdy.sharding",
              mlir::sdy::TensorShardingPerValueAttr::get(ctx, opShardings));

  auto reductionAxisRefs =
      shardyAxisRefsForIndices(reductionAxes, mesh);
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

// Translates every distributed.ManualComputation in `shardyModule` (built by
// CanonicalizeShardedFactorOrderPass's wrapReshapeInLocalManualComputation)
// into a real sdy.manual_computation. This op's
// argument_shardings/output_shardings/ manual_axes are ODS-declared arguments
// (stored under their own plain attribute names, e.g. "argument_shardings" --
// NOT the "distributed."- prefixed discardable attributes
// constructShardyAttributes's generic opsToProcess walk below scans for), so
// this must run as its own pass over the module rather than folding into that
// walk; the two don't interact either way regardless of ordering, since the
// string keys never collide.
static void convertManualComputationsToShardy(
    ModuleOp shardyModule, StringRef meshName,
    const KernelMesh &mesh) {
  MLIRContext *ctx = shardyModule.getContext();

  // Collect first, then convert: converting erases each op and moves its
  // region, so mutating while walking would visit already-invalidated state.
  llvm::SmallVector<DistributedManualComputationOp> manualOps;
  shardyModule.getRegion().walk(
      [&](DistributedManualComputationOp op) { manualOps.push_back(op); });

  for (DistributedManualComputationOp manualOp : manualOps) {
    auto inShardings = translateIndexedShardingPerValue(
        ctx, meshName, manualOp.getArgumentShardings(), mesh);
    auto outShardings = translateIndexedShardingPerValue(
        ctx, meshName, manualOp.getOutputShardings(), mesh);

    llvm::SmallVector<StringAttr> manualAxes;
    manualAxes.reserve(manualOp.getManualAxes().size());
    for (int64_t index : manualOp.getManualAxes()) {
      for (mlir::sdy::AxisRefAttr ref : mesh.slotAxes[index]) {
        manualAxes.push_back(StringAttr::get(ctx, ref.getName()));
      }
    }

    OpBuilder builder(manualOp);
    auto sdyOp = builder.create<mlir::sdy::ManualComputationOp>(
        manualOp.getLoc(), manualOp.getResultTypes(), manualOp.getInputs(),
        inShardings.getShardings(), outShardings.getShardings(), manualAxes);

    sdyOp.getBody().takeBody(manualOp.getBody());

    // sdy.manual_computation requires an sdy.return terminator, not
    // distributed.DistributedYield.
    auto yieldOp =
        cast<DistributedYieldOp>(sdyOp.getBody().front().getTerminator());
    OpBuilder(yieldOp).create<mlir::sdy::ReturnOp>(yieldOp.getLoc(),
                                                   yieldOp.getReturns());
    yieldOp.erase();

    manualOp.replaceAllUsesWith(sdyOp.getResults());
    manualOp.erase();
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
 * blocks passed in from the collectives). This depends on
 * CanonicalizeShardedFactorOrderPass having already put every dimension's
 * sharded factors in major-most position (see the doc comment above
 * runShardyLowering) -- without that precondition this would not hold up to
 * all sharding dimensions.
 */
static void
constructShardyAttributes(DistributedKernelOp originalKernel,
                          ModuleOp shardyModule, const KernelMesh &mesh) {
  MLIRContext *ctx = shardyModule.getContext();
  constexpr llvm::StringLiteral kMeshName = "mesh";

  // Later attributes reference this mesh by the axis names `mesh` assigns to
  // each partitioning-axis slot.
  OpBuilder meshBuilder(ctx);
  meshBuilder.setInsertionPointToStart(shardyModule.getBody());
  meshBuilder.create<mlir::sdy::MeshOp>(
      shardyModule.getLoc(), kMeshName,
      mlir::sdy::MeshAttr::get(ctx, mesh.axes));

  // Must run before the generic per-op walk below builds sdy.sharding_
  // constraint ops against any distributed.ManualComputation operand: a real
  // sdy.manual_computation's own in_shardings already establish its operand's
  // sharding directly, so there's nothing left for that walk to see once this
  // runs (it never looked at this op's attrs to begin with -- see this
  // function's own comment).
  convertManualComputationsToShardy(shardyModule, kMeshName, mesh);

  auto shardyFunc = shardyModule.lookupSymbol<func::FuncOp>("kernel");
  if (!shardyFunc) {
    shardyModule.emitError() << "missing shardy kernel wrapper function";
    return;
  }

  // Need to add attributes for the func arg, return shardings based on the
  // original kernel
  mlir::sdy::TensorShardingPerValueAttr argShardings =
      translateIndexedShardingPerValue(ctx, kMeshName,
                                       originalKernel.getArgumentShardings(),
                                       mesh);
  for (auto [argIndex, blockArg] : llvm::enumerate(shardyFunc.getArguments())) {
    mlir::sdy::setSharding(blockArg, argShardings.getSharding(argIndex));
  }

  mlir::sdy::TensorShardingPerValueAttr outputShardings =
      translateIndexedShardingPerValue(ctx, kMeshName,
                                       originalKernel.getOutputShardings(),
                                       mesh);
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
      auto baseResultShardings = translateIndexedShardingPerValue(
          ctx, kMeshName, outputShardings, mesh);
      auto reductionAxes = computeReductionAxes(
          argumentShardings, outputShardings, mesh);
      insertPlaceholderAllReduces(ctx, kMeshName, op, baseResultShardings,
                                  reductionAxes, mesh);
    }

    if (!argumentShardings) {
      continue;
    }
    OpBuilder opBuilder(op);
    for (auto [operandIndex, indexed] :
         llvm::enumerate(argumentShardings.getShardings())) {
      Value operand = op->getOperand(operandIndex);
      auto sharding =
          translateIndexedSharding(ctx, kMeshName, indexed, mesh);
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

    // Cross-check against the kernel's own operand: that's the external ABI
    // shape (already local, per ClusterDistributedKernels.cpp/
    // InlineDeviceLocalAxesPass) this Shardy-shrunk block-argument type is
    // supposed to land on exactly. Nothing before this point actually
    // verifies the two agree -- ClusterDistributedKernels.cpp divides the
    // operand down once at construction, InlineDeviceLocalAxesPass grows it
    // for DeviceLocal factors, and this function independently shrinks the
    // block argument by the sharded factors -- so a divergence here means
    // that bookkeeping chain is out of sync somewhere upstream. Remark-only,
    // same rationale as checkKernelOperandGrowthConsistency: SearchStrategies
    // .cpp runs this pass internally to score every candidate, so a hard
    // failure here would make the search unable to use any candidate that
    // happens to hit this.
    if (argIndex < kernelOp.getArguments().size()) {
      auto operandType = dyn_cast<RankedTensorType>(
          kernelOp.getArguments()[argIndex].getType());
      if (operandType &&
          operandType.getShape() != ArrayRef<int64_t>(updatedShape)) {
        kernelOp.emitRemark()
            << "distributed-lower-kernels: operand " << argIndex
            << "'s actual type " << operandType
            << " doesn't match the local shape this pass derives from "
               "argument_shardings after sharding ("
            << RankedTensorType::get(updatedShape, rankedType.getElementType())
            << ")";
      }
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
 * <sharding_axes> * serialize_axis, where serialize axis is the contiguousmost
 * (i.e. minor-most). We can have multiple axes on the same tensor dimension
 * (induced from i.e. a reshape), leading to a composite sharding along the
 * lines of <sharding_axes> * serialize_ax * <sharding_axes> * serialize_ax --
 * this doesn't play very nice with Shardy rewrite capabilities, which just
 * takes a number of dims to shard over.
 *
 * CanonicalizeShardedFactorOrderPass is expected to have already run and
 * established that every dimension's sharded (sub-)axes form a major-most
 * prefix with the serialize/local factors an implicit minor-most remainder --
 * i.e. the "<sharding_axes> * serialize_ax * <sharding_axes> * serialize_ax"
 * composite case above never actually reaches this pass with the serialize
 * axes anywhere but the end. That's what justifies treating the axis as
 * either embarrassingly parallel (pass through) or an associative reduction
 * (reassociated, with the serialize axes shifted to the end) and just taking
 * the total parallelism over the axis as a flat shard count below: it's a
 * consequence of the upstream invariant, not an unverified hope about ops
 * that don't parallelize. However, we will still have to be very careful
 * about communication and slicing ops.
 */

/**
 * Invokes Shardy's ConvertGlobalToLocal pass and then drop the mesh/sharding
 * metadata from the synthetic wrapper.
 */
static LogicalResult runShardyLowering(ModuleOp shardyModule, bool verify) {
  PassManager pm(shardyModule.getContext());
  // With maximal logical parallelism (one device id per tensor element),
  // MLIR's verifier can spend a very long time walking the resulting
  // hundred-thousand-plus-element replica-group attributes; skip it here by
  // default (verify defaults to false at the pass level) and only pay that
  // cost when explicitly asked for.
  pm.enableVerifier(verify);
  // Most sharding_constraint ops we insert pin the same sharding an operand
  // already has, so ApplyShardingConstraintsPass folds those away as a
  // no-op. But constructShardyAttributes only inserts a constraint when it
  // actually differs from what the operand already carries (see its own
  // comment), so a constraint surviving past that pass is not a bug: it's a
  // real resharding boundary. ConvertGlobalToLocal has no pattern for
  // sdy.sharding_constraint itself -- it expects such survivors to already
  // be converted to sdy.reshard, which ShardingConstraintToReshardPass does
  // unconditionally for whatever the previous pass left behind.
  pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::sdy::createShardingConstraintToReshardPass());
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
    return failure();
  }
  return success();
}

struct LowerKernelsPass : public impl::LowerKernelsPassBase<LowerKernelsPass> {
  using LowerKernelsPassBase::LowerKernelsPassBase;

  // Shards a single kernel by splitting its partitioning factors into the
  // subset that is safe to lower now and the subset that should remain on the
  // local kernel view. The sharding decision is based on axis kind: logical
  // mesh axes are optionally shardable, while replication and device-local axes
  // are never lowered through Shardy.
  bool lowerKernel(DistributedKernelOp kernelOp, Operation *mainScopeOp,
                   bool lowerLogical) {
    FactorsPerDim shardableParts;
    FactorsPerDim nonShardableParts;
    splitPartitioningAxesByShardability(kernelOp, lowerLogical, shardableParts,
                                        nonShardableParts);
    KernelMesh mesh = buildKernelMesh(kernelOp, shardableParts);

    // The temporary func.func is intentionally a thin wrapper: Shardy works on
    // function-shaped IR, while the kernel op itself carries the local ABI
    // types. Built from the kernel body's own (possibly already
    // partially-shrunk) block-arg/yield types rather than the kernel op's
    // external operand/result types, which only coincide on the first pass
    // invocation and diverge once any factor has been lowered -- using the
    // body's types keeps a second run idempotent.
    auto shardyModuleOrFailure = buildKernelBodyModule(kernelOp);
    if (failed(shardyModuleOrFailure)) {
      return false;
    }
    ModuleOp shardyModule = *shardyModuleOrFailure;
    constructShardyAttributes(kernelOp, shardyModule, mesh);
    // debug logging option
    if (dumpShardyModules) {
      llvm::dbgs() << "Dumping Shardy module:\n";
      llvm::dbgs() << shardyModule << "\n";
    }
    // Transform and copy back the module
    bool lowered =
        succeeded(runShardyLowering(shardyModule, verifyShardyLowering));
    stripPlaceholderAllReduces(shardyModule);
    if (dumpLoweredModules) {
      llvm::dbgs() << "Dumping lowered module:\n";
      llvm::dbgs() << shardyModule << "\n";
    }
    // Still splice back and continue bookkeeping even on failure, matching
    // the rest of the pass's remark-and-continue style: this keeps the walk
    // below able to report every failing kernel in one run instead of just
    // the first, and dumpLoweredModules still shows the (partially) lowered
    // body. The pass-wide failure signaled below is what actually stops this
    // from being mistaken for a successful lowering, e.g. by the search's
    // own scoring, which treats a failed pipeline run as a candidate to
    // reject rather than a valid, silently-broken one to keep.
    copyShardyModuleToKernelAndErase(shardyModule, kernelOp);
    updateKernelArgumentTypes(kernelOp, shardableParts);
    removeShardedFactorsFromPartitioningAxes(kernelOp, shardableParts,
                                             nonShardableParts);

    (void)mainScopeOp;
    return lowered;
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

    bool allLowered = true;
    module_op.walk([&](DistributedKernelOp kernel_op) {
      if (!lowerKernel(kernel_op, mainScopeOp, lowerLogicalAxes)) {
        allLowered = false;
      }
    });
    if (!allLowered) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
