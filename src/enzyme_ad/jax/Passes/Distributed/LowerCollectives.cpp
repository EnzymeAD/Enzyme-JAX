#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveDecomposer.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/MeshCostMetadata.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/CollectivePrimitiveBuilders.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_LOWERCOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Replaces `plan`'s distributed.Collective/Await pair with the real
// StableHLO async op chain its decomposition calls for: one
// async_start/async_done pair per PrimitiveStep
// (CollectivePrimitiveBuilders.h), each step's result threaded into the next
// step's operand (CollectivePrimitiveBuilders.h's flat-1-D convention, since a
// PrimitiveStep carries no tensor shape; see that header's file comment).
// An empty chain (the collective moves nothing) is a no-op alias straight
// from the input to the result, matching ScheduleGraph.cpp's
// buildCollectiveNodes.
void lowerCollectivePlan(const CollectivePlan &plan,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                         OpBuilder &builder) {
  DistributedCollectiveOp collective = plan.collective;
  DistributedAwait await = plan.await;
  Location loc = collective.getLoc();

  if (plan.chain.empty()) {
    await.getValue().replaceAllUsesWith(plan.inputPort);
    await.erase();
    collective.erase();
    return;
  }

  builder.setInsertionPoint(collective);

  // Every AllReduce/ReduceScatter step in the chain shares one reduction
  // semantics (CollectiveDecomposer.cpp asserts every reduction_bodies
  // region classifies to the same ReductionKind), so the first reduction
  // body stands in for all of them.
  Region *reductionBody = collective.getReductionBodies().empty()
                              ? nullptr
                              : &collective.getReductionBodies()[0];

  Value current = flattenTo1D(builder, loc, plan.inputPort);
  for (const PrimitiveStep &step : plan.chain) {
    switch (step.kind) {
    case PrimitiveKind::AllReduce:
      assert(reductionBody && "an AllReduce step implies a reduction group");
      current = buildAllReduceStep(builder, loc, current, step, meshAxisTypes,
                                   *reductionBody);
      break;
    case PrimitiveKind::ReduceScatter:
      assert(reductionBody && "a ReduceScatter step implies a reduction group");
      current = buildReduceScatterStep(builder, loc, current, step,
                                       meshAxisTypes, *reductionBody);
      break;
    case PrimitiveKind::AllGather:
      current = buildAllGatherStep(builder, loc, current, step, meshAxisTypes);
      break;
    case PrimitiveKind::AllToAll:
      current = buildAllToAllStep(builder, loc, current, step, meshAxisTypes);
      break;
    case PrimitiveKind::Permute:
      current = buildPermuteStep(builder, loc, current, step, meshAxisTypes,
                                 collective);
      break;
    case PrimitiveKind::LocalSlice:
      current = buildLocalSliceStep(builder, loc, current, step, meshAxisTypes);
      break;
    }
  }

  Value result =
      reshapeFrom1D(builder, loc, current,
                    cast<RankedTensorType>(await.getValue().getType()));
  await.getValue().replaceAllUsesWith(result);
  await.erase();
  collective.erase();
}

struct LowerCollectivesPass
    : public impl::LowerCollectivesPassBase<LowerCollectivesPass> {
  using LowerCollectivesPassBase::LowerCollectivesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    if (failed(physicalMesh)) {
      // findUniquePhysicalMesh already emitted a diagnostic.
      signalPassFailure();
      return;
    }
    SmallVector<PhysicalCommAxisType> meshAxisTypes;
    for (Attribute axisAttr : physicalMesh->getAxesAttr())
      meshAxisTypes.push_back(
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));
    MeshCostParams params = meshCostParamsFromPhysicalMesh(*physicalMesh);

    // Collect before mutating: lowering a plan erases its Collective/Await
    // pair, which would invalidate an in-flight walk.
    SmallVector<DistributedCollectiveOp> collectives;
    module.walk([&](DistributedCollectiveOp collective) {
      collectives.push_back(collective);
    });

    OpBuilder builder(&getContext());
    for (DistributedCollectiveOp collective : collectives) {
      std::string failureReason;
      std::optional<CollectivePlan> plan = planCollective(
          collective, meshAxisTypes, params, failureReason, PlanOptions{});
      if (!plan) {
        // This pass is meant to run winner-only, on a single search
        // candidate that has already been validated (planned successfully)
        // during scoring, so an unplannable collective here means an
        // unexpected precondition violation, not a normal outcome to skip
        // over quietly.
        collective.emitError()
            << "distributed-lower-collectives: could not plan collective ("
            << failureReason << ")";
        signalPassFailure();
        return;
      }
      lowerCollectivePlan(*plan, meshAxisTypes, builder);
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
