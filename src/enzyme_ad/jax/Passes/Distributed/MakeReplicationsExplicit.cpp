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

// Builds a fresh factor-group covering the whole module mesh: one
// full-extent, unit-stride factor per physical axis. This is the minuend
// axis::subtractSpace needs to compute what a mesh operand is missing --
// PhysicalCommAxisType's equivalence is defined structurally (same physical
// mesh symbol + same axis index, see PhysicalCommAxisType::equivalent), not
// by SSA identity, so a brand new distributed.GetPhysicalMeshAxes here is
// automatically recognized as the same axis space any other collective's
// mesh operands were built from -- no CSE required.
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

// A DistributedCollective's input_mesh/output_mesh/mapping are only required
// (by the op's own verifier) to fully account for whichever physical axes
// they actually mention -- a physical axis absent from input_mesh entirely
// is well-defined (the value is uniformly replicated across it) but that
// meaning is only recoverable by comparing against the whole module's
// PhysicalMesh from outside the collective itself. This rewrites each such
// collective so every physical axis of the module's mesh is always directly
// represented as either a real spatial factor or an explicit ReplicationAxis
// factor, on both the input/output mesh operands and in `mapping`, so no
// downstream consumer of a collective needs to infer replication from
// omission.
//
// reduction_groups/mapping never need their own separate "missing axis"
// scan: MaterializeDistributedCollectives.cpp (the sole producer of these
// operands) always builds a collective's input_mesh to already include its
// own reduction_groups' axes, and builds mapping's lhs/rhs FactorGroups from
// that same producer/consumer partitioning-axis basis -- so any physical
// axis reachable through reduction_groups or mapping is already reachable
// through input_mesh/output_mesh directly, and checking those two mesh
// operands alone is exhaustive.
struct MakeCollectiveReplicationsExplicit
    : public OpRewritePattern<DistributedCollectiveOp> {
  MakeCollectiveReplicationsExplicit(MLIRContext *context,
                                     PhysicalMeshOp physicalMesh)
      : OpRewritePattern(context), physicalMesh(physicalMesh) {}

  LogicalResult matchAndRewrite(DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Cheap early exit computed with ZERO IR mutation (no op creation at
    // all -- fullExtentOfMesh reads physicalMesh's own type attrs
    // directly, and axis::getFactorGroupExtent is a pure query) so a
    // "nothing to do" match attempt never itself changes the IR before
    // even reaching buildFullMeshFactorGroup below. This matters for more
    // than a style nicety: OpRewritePattern's contract requires a
    // failure() return to have made no IR changes, and
    // buildFullMeshFactorGroup creates real ops (a fresh
    // GetPhysicalMeshAxesOp + factors + product) -- calling it
    // unconditionally on every match attempt, including already-fixed
    // collectives, would leave dangling ops behind on every "no-op" visit
    // and confuse the greedy driver's convergence tracking. This check is
    // a plain necessary condition (an operand at full extent might still
    // be composed differently than expected), not a substitute for the
    // real axis::subtractSpace-based check below -- it only exists to
    // avoid mutating the IR before that real check has had a chance to
    // run cheaply.
    uint64_t fullExtentOfMesh = 1;
    // matchAndRewrite is const, so the `physicalMesh` member is accessed as
    // const here; getAxesAttr() isn't const-qualified (ODS-generated op
    // accessors aren't), so a plain non-const copy is needed to call it.
    PhysicalMeshOp mesh = physicalMesh;
    for (Attribute axisAttr : mesh.getAxesAttr()) {
      fullExtentOfMesh *=
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue())
              .getExtent();
    }
    FailureOr<uint64_t> inputExtent =
        axis::getFactorGroupExtent(op.getInputMesh());
    FailureOr<uint64_t> outputExtent =
        axis::getFactorGroupExtent(op.getOutputMesh());
    assert(succeeded(inputExtent) && succeeded(outputExtent) &&
          "collective mesh operand must have a computable factor-group "
          "extent");
    if (*inputExtent == fullExtentOfMesh && *outputExtent == fullExtentOfMesh) {
      return failure();
    }

    // Every axis-algebra op below is pure metadata and belongs at module
    // scope, mirroring the ModuleScopeGuard convention used throughout the
    // rest of this dialect's lowering passes.
    axis::ModuleScopeGuard moduleScope(rewriter);
    TV_FactorGroup fullMesh = buildFullMeshFactorGroup(physicalMesh, rewriter, loc);

    auto inputFactors = axis::getProductProvenanceFactors(op.getInputMesh());
    auto outputFactors = axis::getProductProvenanceFactors(op.getOutputMesh());
    // Guaranteed by the collective having reached this pass at all: its own
    // mesh operands are always axis.product results (see
    // MaterializeDistributedCollectives.cpp, the sole producer of these
    // operands).
    assert(succeeded(inputFactors) && succeeded(outputFactors) &&
          "collective mesh operand must be produced by axis.product");

    FailureOr<SmallVector<TV_AxisFactor>> missingInput =
        axis::subtractSpace(fullMesh, *inputFactors, rewriter);
    FailureOr<SmallVector<TV_AxisFactor>> missingOutput =
        axis::subtractSpace(fullMesh, *outputFactors, rewriter);
    assert(succeeded(missingInput) && succeeded(missingOutput) &&
          "module physical mesh axes must be representable as factors of "
          "a collective's own mesh operands");

    if (missingInput->empty() && missingOutput->empty()) {
      return failure();
    }

    SmallVector<TV_AxisFactor> newInputFactors(*inputFactors);
    SmallVector<TV_AxisFactor> newOutputFactors(*outputFactors);
    auto mapOp = cast<axis::AxisMapOp>(op.getMapping().getDefiningOp());
    SmallVector<TV_FactorGroup> newMappingLhs = mapOp.getTypedMappingLhs();
    SmallVector<TV_FactorGroup> newMappingRhs = mapOp.getTypedMappingRhs();

    // missingInput and missingOutput are independent sets -- a physical
    // axis can be missing from one mesh operand without being missing from
    // the other (or from neither, or from both), and each direction gets
    // its own fix rather than trying to detect/pair an axis that happens
    // to be missing on both sides: that degenerate case is correctly (and
    // deliberately -- see below) covered by simply letting both loops run
    // independently over it, each contributing its own pair.
    //
    // Per DistributedCollectiveOp::verify() (CollectiveOps.cpp): the
    // "counted" side of a mapping pair (matched against its own mesh
    // operand's factor list) must carry the REAL missing factor, since a
    // mesh operand's own factors are compared unfiltered; only a mapping
    // pair's *other*, uncounted side (filtered out via
    // filterOutReplicationFactors before that comparison) may safely be a
    // synthetic Replicate that is never folded into any mesh operand. Get
    // this backwards (e.g. fold a fresh Replicate into a mesh operand
    // instead of the real factor) and the verifier's index-space-equality
    // check fails, since the mesh operand side would count a factor the
    // mapping side doesn't.
    //
    // Missing from output_mesh (RHS) means the result is replicated over
    // that axis: fold the real factor into output_mesh (counted side) and
    // pair it against a fresh, never-mesh-folded Replicate standing in for
    // "no real axis here" on mapping's lhs (uncounted side).
    for (TV_AxisFactor missing : *missingOutput) {
      unsigned extent = axis::getFactorExtent(missing);
      auto replAxis = rewriter.create<ReplicationAxisOp>(loc, extent);
      TV_AxisFactor replicateFactor =
          axis::viewAxisAsFactor(replAxis.getAxis(), rewriter, loc);
      newOutputFactors.push_back(missing);
      newMappingLhs.push_back(
          axis::viewFactorsAsProduct(replicateFactor, rewriter, loc));
      newMappingRhs.push_back(
          axis::viewFactorsAsProduct(missing, rewriter, loc));
    }
    // Symmetric: missing from input_mesh (LHS) means the input is
    // replicated over that axis -- real factor folded into input_mesh
    // (counted side, lhs of the new pair), fresh Replicate confined to
    // mapping's rhs (uncounted side), never folded into output_mesh.
    for (TV_AxisFactor missing : *missingInput) {
      unsigned extent = axis::getFactorExtent(missing);
      auto replAxis = rewriter.create<ReplicationAxisOp>(loc, extent);
      TV_AxisFactor replicateFactor =
          axis::viewAxisAsFactor(replAxis.getAxis(), rewriter, loc);
      newInputFactors.push_back(missing);
      newMappingLhs.push_back(
          axis::viewFactorsAsProduct(missing, rewriter, loc));
      newMappingRhs.push_back(
          axis::viewFactorsAsProduct(replicateFactor, rewriter, loc));
    }

    TV_FactorGroup newInputMesh =
        axis::viewFactorsAsProduct(newInputFactors, rewriter, loc);
    TV_FactorGroup newOutputMesh =
        axis::viewFactorsAsProduct(newOutputFactors, rewriter, loc);

    rewriter.modifyOpInPlace(op, [&] {
      op.getInputMeshMutable().assign(newInputMesh);
      op.getOutputMeshMutable().assign(newOutputMesh);
    });
    rewriter.modifyOpInPlace(mapOp, [&] {
      mapOp.getMappingLhsMutable().assign(
          SmallVector<Value>(newMappingLhs.begin(), newMappingLhs.end()));
      mapOp.getMappingRhsMutable().assign(
          SmallVector<Value>(newMappingRhs.begin(), newMappingRhs.end()));
    });
    return success();
  }

private:
  PhysicalMeshOp physicalMesh;
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

    // Keep `mapping`'s factor lists in maximal one-to-one form after adding
    // the new whole-axis Replicate<->Replicate pairs above, matching the
    // canonical form DropIdentityCollectivesPass also restores after its own
    // rewrite.
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
