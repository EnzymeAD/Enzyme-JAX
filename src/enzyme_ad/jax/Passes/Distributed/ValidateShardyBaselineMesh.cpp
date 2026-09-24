#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_VALIDATESHARDYBASELINEMESHPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Checks that every sdy.mesh axis is a contiguous device-id window of the
// physical mesh's factorization, so Shardy's per-axis shardings can be read as
// physical axis factors. Runs at pipeline entry because the user-supplied mesh
// declarations are the only input to check. On success it stamps the module
// with kShardyBaselineAttr, which switches every later
// ShardyLogicalAxisAnalysis into Shardy baseline mode and stays on the module
// to mark it as a baseline.
struct ValidateShardyBaselineMeshPass
    : public impl::ValidateShardyBaselineMeshPassBase<
          ValidateShardyBaselineMeshPass> {
  using ValidateShardyBaselineMeshPassBase::ValidateShardyBaselineMeshPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    SmallVector<PhysicalMeshOp> physicalMeshes;
    SmallVector<sdy::MeshOp> shardyMeshes;
    for (Operation &op : moduleOp.getBody()->getOperations()) {
      if (auto physical = dyn_cast<PhysicalMeshOp>(&op))
        physicalMeshes.push_back(physical);
      else if (auto shardy = dyn_cast<sdy::MeshOp>(&op))
        shardyMeshes.push_back(shardy);
    }

    if (physicalMeshes.size() != 1) {
      moduleOp.emitError()
          << "preserve-shardy-sharding requires exactly one "
             "distributed.PhysicalMesh (run --insert-physical-mesh first), "
             "found "
          << physicalMeshes.size();
      return signalPassFailure();
    }

    SmallVector<PhysicalAxisSpec> physicalAxes;
    for (Attribute axisAttr : physicalMeshes.front().getAxesAttr()) {
      auto axis =
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue());
      physicalAxes.push_back({static_cast<int64_t>(axis.getExtent()),
                              static_cast<int64_t>(axis.getIdStride())});
    }

    for (sdy::MeshOp shardyMesh : shardyMeshes) {
      if (!shardyMesh.getMesh().getDeviceIds().empty()) {
        shardyMesh.emitError()
            << "preserve-shardy-sharding requires sdy.mesh @"
            << shardyMesh.getSymName() << " to use the default (iota) device "
            << "order";
        return signalPassFailure();
      }
      auto projection =
          projectShardyMesh(shardyMesh.getMesh().getAxes(), physicalAxes);
      if (!projection) {
        shardyMesh.emitError()
            << "sdy.mesh @" << shardyMesh.getSymName()
            << " cannot be projected onto distributed.PhysicalMesh @"
            << physicalMeshes.front().getSymName() << ": "
            << llvm::toString(projection.takeError());
        return signalPassFailure();
      }
    }
    moduleOp->setAttr(kShardyBaselineAttr, UnitAttr::get(&getContext()));
  }
};

} // namespace
} // namespace mlir::enzyme::distributed
