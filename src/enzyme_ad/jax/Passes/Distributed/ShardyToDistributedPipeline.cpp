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
};
} // namespace

void registerShardyToDistributedPipeline() {
  PassPipelineRegistration<ShardyToDistributedPipelineOptions>(
      "shardy-to-distributed-pipeline",
      "Run the Shardy-to-distributed conversion pipeline from explicit "
      "reshards through kernel clustering. End result: all computation "
      "distributed on maximal logical axes",
      [](OpPassManager &pm, const ShardyToDistributedPipelineOptions &options) {
        pm.addPass(mlir::sdy::createInsertExplicitReshardsPass());
        pm.addPass(createConvertMainToDistributedFunctionPass());
        MaterializeDistributedCollectivesPassOptions materializeOptions;
        materializeOptions.dumpValueAxes = options.dumpValueAxes;
        materializeOptions.dumpOperationAxes = options.dumpOperationAxes;
        pm.addPass(
            createMaterializeDistributedCollectivesPass(materializeOptions));
        pm.addPass(mlir::sdy::createDropShardingAndMeshPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createClusterDistributedKernelsPass());
      });
}

} // namespace mlir::enzyme::distributed
