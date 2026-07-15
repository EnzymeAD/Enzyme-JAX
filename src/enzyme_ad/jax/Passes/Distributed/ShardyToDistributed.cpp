#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/FindShardyFunctionsAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct ShardyToDistributedPass
    : public impl::ShardyToDistributedPassBase<ShardyToDistributedPass> {
  using ShardyToDistributedPassBase::ShardyToDistributedPassBase;

  // For now we simplify our life by requiring all shardy functions to have the
  // same mesh. This will be relaxed at need as implementation progresses.
  sdy::MeshAttr getCommonMesh(const FindShardyFunctionsAnalysis &analysis) {
    auto shardyFunctions = analysis.getShardyFunctions();
    sdy::MeshAttr commonMesh = nullptr;
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         shardyFunctions) {
      if (info.meshes.size() != 1) {
        getOperation()->emitError()
            << "expected shardy function to have exactly one mesh, found "
            << info.meshes.size();
        signalPassFailure();
        return nullptr;
      }
      if (!commonMesh) {
        commonMesh = info.meshes[0];
      } else if (commonMesh != info.meshes[0]) {
        getOperation()->emitError()
            << "expected all shardy functions to have the same mesh, found "
            << commonMesh << " and " << info.meshes[0];
        signalPassFailure();
        return nullptr;
      }
    }
    return commonMesh;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    FailureOr<distributed::PhysicalMeshOp> physicalMesh =
        distributed::findUniquePhysicalMesh(moduleOp);
    if (failed(physicalMesh)) {
      signalPassFailure();
      return;
    }

    const FindShardyFunctionsAnalysis &analysis =
        getAnalysis<FindShardyFunctionsAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }
    if (analysis.getShardyFunctions().empty()) {
      moduleOp.emitRemark() << "no shardy functions found, skipping pass";
      return;
    }

    sdy::MeshAttr commonMesh = getCommonMesh(analysis);
    if (!commonMesh) {
      return;
    }

    // Create a new mesh computation using the modules pysical mesh
  OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointAfter(
    *physicalMesh); // graph region, doesn't matter
    distributed::MeshComputationOp meshComputation =
        builder.create<distributed::MeshComputationOp>(
      moduleOp.getLoc(), builder.getStringAttr("mesh_computation"),
      FlatSymbolRefAttr::get(physicalMesh->getSymNameAttr()));
    Region &meshComputationBody = meshComputation.getBody();
    if (meshComputationBody.empty()) {
      meshComputationBody.emplaceBlock();
    }
    builder.setInsertionPointToStart(&meshComputationBody.front());

    // Create a new logical mesh inside of the mesh computation using the common
    // shardy mesh for axis sizes. Internally record a mapping from shardy axis
    // names to distributed axis values.
    llvm::SmallVector<int32_t> logicalAxisExtents;
    llvm::SmallVector<StringAttr> shardyAxisNames;
    for (sdy::MeshAxisAttr axis : commonMesh.getAxes()) {
      int64_t axisSize = axis.getSize();
      if (axisSize <= 0 || axisSize > std::numeric_limits<int32_t>::max()) {
        moduleOp.emitError() << "unsupported shardy mesh axis size " << axisSize
                             << " for axis " << axis.getName();
        signalPassFailure();
        return;
      }
      logicalAxisExtents.push_back(static_cast<int32_t>(axisSize));
      shardyAxisNames.push_back(builder.getStringAttr(axis.getName()));
    }

    distributed::LogicalMeshAxesOp logicalMeshAxes =
        builder.create<distributed::LogicalMeshAxesOp>(
        moduleOp.getLoc(), builder.getDenseI32ArrayAttr(logicalAxisExtents));

    llvm::DenseMap<StringAttr, Value> shardyToDistributedAxis;
    for (auto [idx, distributedAxis] :
         llvm::enumerate(logicalMeshAxes.getAxes())) {
      shardyToDistributedAxis[shardyAxisNames[idx]] = distributedAxis;
    }

    (void)shardyToDistributedAxis;
    (void)logicalMeshAxes;
    (void)meshComputation;
  }
};

} // namespace

} // namespace mlir::enzyme::distributed