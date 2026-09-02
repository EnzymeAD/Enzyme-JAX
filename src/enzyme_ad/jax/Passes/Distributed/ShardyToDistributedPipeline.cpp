#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"

namespace mlir::enzyme::distributed {

void registerShardyToDistributedPipeline() {
  PassPipelineRegistration<>(
      "shardy-to-distributed-pipeline",
      "Run the Shardy-to-distributed conversion pipeline from explicit "
      "reshards through kernel clustering. End result: all computation "
      "distributed on maximal logical axes",
      [](OpPassManager &pm) {
        pm.addPass(mlir::sdy::createInsertExplicitReshardsPass());
        pm.addPass(createConvertMainToDistributedFunctionPass());
        pm.addPass(createMaterializeDistributedCollectivesPass());
        pm.addPass(mlir::sdy::createDropShardingAndMeshPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createClusterDistributedKernelsPass());
      });
}

} // namespace mlir::enzyme::distributed
