#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"

namespace mlir::enzyme::distributed {

namespace {
// Pipeline-level mirror of MaterializeDistributedCollectivesPass's dump
// options, forwarded to that pass when constructing the pipeline.
struct ShardyToDistributedPipelineOptions
    : public PassPipelineOptions<ShardyToDistributedPipelineOptions> {
  Option<bool> dumpValueAxes{
      *this, "dump-value-axes",
      llvm::cl::desc("Print ShardyLogicalAxisAnalysis axis symbols for each "
                     "SSA value in main, before collective materialization"),
      llvm::cl::init(false)};
  Option<bool> dumpOperationAxes{
      *this, "dump-operation-axes",
      llvm::cl::desc("Print ShardyLogicalAxisAnalysis axis symbols for each "
                     "operation in main, before collective materialization"),
      llvm::cl::init(false)};
  Option<bool> preserveShardySharding{
      *this, "preserve-shardy-sharding",
      llvm::cl::desc("Lower the Shardy plan itself as the expert baseline: "
                     "the sdy.sharding attributes bind each value's axes "
                     "instead of the search. Requires a "
                     "distributed.PhysicalMesh onto which every sdy.mesh "
                     "projects (run --insert-physical-mesh first)"),
      llvm::cl::init(false)};
};
} // namespace

void registerShardyToDistributedPipeline() {
  PassPipelineRegistration<ShardyToDistributedPipelineOptions>(
      "shardy-to-distributed-pipeline",
      "Run the Shardy-to-distributed conversion pipeline from explicit "
      "reshards through kernel clustering. End result: all computation "
      "distributed on maximal logical axes",
      [](OpPassManager &pm, const ShardyToDistributedPipelineOptions &options) {
        if (options.preserveShardySharding)
          pm.addPass(createValidateShardyBaselineMeshPass());
        pm.addPass(mlir::sdy::createInsertExplicitReshardsPass());
        // Merge duplicated computation (e.g. per-layer masks, per-projection
        // tables) before materialization binds each copy to its own axes.
        // Only copies with equal sdy.sharding merge. Runs before the
        // gather/scatter isolation so CSE cannot fold two isolated ops.
        pm.addPass(createCSEPass());
        // The baseline lowers Shardy's own plan, which these breaks don't
        // carry shardings for.
        if (!options.preserveShardySharding)
          pm.addPass(createIsolateGatherScatterPass());
        pm.addPass(createConvertMainToDistributedFunctionPass());
        MaterializeDistributedCollectivesPassOptions materializeOptions;
        materializeOptions.dumpValueAxes = options.dumpValueAxes;
        materializeOptions.dumpOperationAxes = options.dumpOperationAxes;
        pm.addPass(
            createMaterializeDistributedCollectivesPass(materializeOptions));
        pm.addPass(createCanonicalizerPass());
        // Materialization has laid out each copy of duplicated computation
        // separately, and only their sdy.sharding tells apart copies with
        // different layouts, so the shardings must stay until after CSE.
        pm.addPass(createCSEPass());
        // In baseline mode, clustering's axis analysis reads the shardings.
        pm.addPass(createClusterDistributedKernelsPass());
        pm.addPass(mlir::sdy::createDropShardingAndMeshPass());
        pm.addPass(createDistributedVerifyKernelsPass());
      });
}

} // namespace mlir::enzyme::distributed
