#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_MAKEREPLICATIONSEXPLICITPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

using TV_AxisFactor = TypedValue<axis::AxisFactorType>;
using TV_FactorGroup = TypedValue<axis::FactorGroupType>;

// Builds a fresh factor-group covering the whole module mesh, so we can
// use it as the device universe to subtract from later.
TV_FactorGroup buildFullMeshFactorGroup(PhysicalMeshOp physicalMesh,
                                        OpBuilder &builder, Location loc) {
  SmallVector<Type> axisTypes;
  for (Attribute axisAttr : physicalMesh.getAxesAttr()) {
    axisTypes.push_back(cast<TypeAttr>(axisAttr).getValue());
  }
  auto getAxesOp = builder.create<GetPhysicalMeshAxesOp>(
      loc, axisTypes, FlatSymbolRefAttr::get(physicalMesh.getSymNameAttr()));
  auto factors = axis::viewAxesAsFactors(getAxesOp.getAxes(), builder, loc);
  return axis::viewFactorsAsProduct(factors, builder, loc);
}

// Gets the extent just of the physical axes, ignoring any other axes (tensor,
// replication, etc)
static uint64_t getPhysicalAxesExtent(TV_FactorGroup group) {
  auto factors = axis::getProductProvenanceFactors(group);
  uint64_t extent = 1;
  for (TV_AxisFactor factor : *factors) {
    auto provenance = axis::getFactorProvenanceAxis(factor);
    if (isa<PhysicalCommAxisType>(provenance->getType())) {
      extent *= axis::getFactorExtent(factor);
    }
  }
  return extent;
}
// A kernel, collective, or cast using only a subset of the module's
// physical mesh axes is implicitly replicated accross the missing axes.
// This pass adds explicit replication factors to the collectives. Note: a
// missing axis means the DATA is replicated, not necessarily the whole
// collective is replicated: an all-gather may have MeshAx --> Replicate(),
// indicating the output data on MeshAx is replicated but there is still
// only one gather collective.
// In the case that a collective is indeed replicated completely
// over a missing axis (i.e., input and output meshes are missing the same
// factor), emit Ax1 --> Replicate() and Replicate() --> Ax1 instead of Ax1 -->
// Ax1: provides slightly more information.
struct MakeCollectiveReplicationsExplicit
    : public OpRewritePattern<DistributedCollectiveOp> {
  MakeCollectiveReplicationsExplicit(MLIRContext *context,
                                     PhysicalMeshOp physicalMesh)
      : OpRewritePattern(context), physicalMesh(physicalMesh),
        fullExtentOfMesh(physicalMesh.getDeviceCount()) {}

  LogicalResult matchAndRewrite(DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Cheap early exit computed with ZERO IR mutation: check if both the
    // input, output mesh operands already cover the full extent of the
    // physical mesh.
    uint64_t inputExtent = getPhysicalAxesExtent(op.getInputMesh());
    uint64_t outputExtent = getPhysicalAxesExtent(op.getOutputMesh());
    if (inputExtent == fullExtentOfMesh && outputExtent == fullExtentOfMesh) {
      return failure(); // No match, nothing to do.
    }

    // Parallel arrays: addInputMapFactors[i]/addOutputMapFactors[i] is one
    // new mapping pair's (lhs, rhs) contribution -- exactly one side of each
    // pair is a real missing physical factor, the other a same-extent
    // Replicate standing in for "no real axis here" (see the per-side
    // comment below for which side is which and why).
    llvm::SmallVector<TV_AxisFactor> addInputMapFactors;
    llvm::SmallVector<TV_AxisFactor> addOutputMapFactors;

    // Not parallel arrays: just the missing mesh space to add to each mesh
    // operand.
    llvm::SmallVector<TV_AxisFactor> missingInputMeshFactors;
    llvm::SmallVector<TV_AxisFactor> missingOutputMeshFactors;

    // Every axis-algebra op below is pure metadata and belongs at module
    // scope, mirroring the ModuleScopeGuard convention used throughout the
    // rest of this dialect's lowering passes -- built directly at module
    // scope from the start, so (unlike a TemporaryOpGuard) nothing here
    // needs to be individually kept alive past this function.
    axis::ModuleScopeGuard moduleScope(rewriter);

    TV_FactorGroup fullMesh =
        buildFullMeshFactorGroup(physicalMesh, rewriter, loc);

    // Adds the missing physical factors to the list for the mesh operand,
    // and adds the physical axis / replicat factor pairs to the map parallel
    // lists.
    auto findExtraFactors =
        [&fullMesh, &rewriter,
         loc](TV_FactorGroup collectiveMesh,
              llvm::SmallVector<TV_AxisFactor> &missingMeshList,
              llvm::SmallVector<TV_AxisFactor> &missingMapOurs,
              llvm::SmallVector<TV_AxisFactor> &missingMapOther) {
          auto collectiveMeshFactors =
              axis::getProductProvenanceFactors(collectiveMesh);
          assert(succeeded(collectiveMeshFactors) &&
                 "collective mesh operand must be produced by axis.product");
          auto missingPhysicalFactors =
              axis::subtractSpace(fullMesh, *collectiveMeshFactors, rewriter);
          assert(succeeded(missingPhysicalFactors) &&
                 "module physical mesh axes must be representable as "
                 "factors of a collective's own mesh operand");
          llvm::SmallVector<TV_AxisFactor> correspondingReplicationFactors;
          for (TV_AxisFactor factor : *missingPhysicalFactors) {
            // Just need something with the same extent.
            auto replAxis = rewriter.create<ReplicationAxisOp>(
                loc, axis::getFactorExtent(factor));
            correspondingReplicationFactors.push_back(
                axis::viewAxisAsFactor(replAxis.getAxis(), rewriter, loc));
          }

          missingMeshList.append(missingPhysicalFactors->begin(),
                                 missingPhysicalFactors->end());
          missingMapOurs.append(missingPhysicalFactors->begin(),
                                missingPhysicalFactors->end());
          missingMapOther.append(correspondingReplicationFactors.begin(),
                                 correspondingReplicationFactors.end());
        };
    findExtraFactors(op.getInputMesh(), missingInputMeshFactors,
                     addInputMapFactors, addOutputMapFactors);
    findExtraFactors(op.getOutputMesh(), missingOutputMeshFactors,
                     addOutputMapFactors, addInputMapFactors);

    // Extend the collective's own mesh operands with whatever they were
    // missing.
    auto inputFactors = axis::getProductProvenanceFactors(op.getInputMesh());
    auto outputFactors = axis::getProductProvenanceFactors(op.getOutputMesh());
    // Guaranteed by the collective having reached this pass at all: its own
    // mesh operands are always axis.product results (see
    // MaterializeDistributedCollectives.cpp, the sole producer of these
    // operands).
    assert(succeeded(inputFactors) && succeeded(outputFactors) &&
           "collective mesh operand must be produced by axis.product");
    SmallVector<TV_AxisFactor> newInputFactors(*inputFactors);
    newInputFactors.append(missingInputMeshFactors.begin(),
                           missingInputMeshFactors.end());
    SmallVector<TV_AxisFactor> newOutputFactors(*outputFactors);
    newOutputFactors.append(missingOutputMeshFactors.begin(),
                            missingOutputMeshFactors.end());
    TV_FactorGroup newInputMesh =
        axis::viewFactorsAsProduct(newInputFactors, rewriter, loc);
    TV_FactorGroup newOutputMesh =
        axis::viewFactorsAsProduct(newOutputFactors, rewriter, loc);

    // Bundle everything missing into one new mapping pair rather than one
    // pair per factor -- axis::getProductProvenanceFactors and friends can
    // always decompose a multi-factor group back into its individual
    // factors later, so nothing downstream needs these split apart here.
    TV_FactorGroup newLhsGroup =
        axis::viewFactorsAsProduct(addInputMapFactors, rewriter, loc);
    TV_FactorGroup newRhsGroup =
        axis::viewFactorsAsProduct(addOutputMapFactors, rewriter, loc);

    // Build a brand new axis.map rather than mutating the existing one in
    // place: the same axis.map value may be shared by other collectives,
    // which must keep seeing their own original mapping unchanged.
    auto mapOp = cast<axis::AxisMapOp>(op.getMapping().getDefiningOp());
    SmallVector<TV_FactorGroup> oldMappingLhs = mapOp.getTypedMappingLhs();
    SmallVector<TV_FactorGroup> oldMappingRhs = mapOp.getTypedMappingRhs();
    SmallVector<Value> newMappingLhs(oldMappingLhs.begin(),
                                     oldMappingLhs.end());
    SmallVector<Value> newMappingRhs(oldMappingRhs.begin(),
                                     oldMappingRhs.end());
    newMappingLhs.push_back(newLhsGroup);
    newMappingRhs.push_back(newRhsGroup);
    auto newMapOp = rewriter.create<axis::AxisMapOp>(
        loc, ValueRange(newMappingLhs), ValueRange(newMappingRhs));

    rewriter.modifyOpInPlace(op, [&] {
      op.getInputMeshMutable().assign(newInputMesh);
      op.getOutputMeshMutable().assign(newOutputMesh);
      op.getMappingMutable().assign(newMapOp.getMap());
    });
    return success();
  }

private:
  PhysicalMeshOp physicalMesh;
  uint64_t fullExtentOfMesh;
};

struct MakeReplicationsExplicitPass
    : public impl::MakeReplicationsExplicitPassBase<
          MakeReplicationsExplicitPass> {
  using MakeReplicationsExplicitPassBase::MakeReplicationsExplicitPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    if (failed(physicalMesh)) {
      // findUniquePhysicalMesh already emitted a diagnostic.
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);
    patterns.add<MakeCollectiveReplicationsExplicit>(context, *physicalMesh);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Restore mappings to their decomposed form
    PassManager pm(context);
    pm.addPass(createCanonicalizeAxisMapsPass());
    if (failed(pm.run(module))) {
      module.emitError() << "canonicalize-axis-maps sub-pass failed";
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
