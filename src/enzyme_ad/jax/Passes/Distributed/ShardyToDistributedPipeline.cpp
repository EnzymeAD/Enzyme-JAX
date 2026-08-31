#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace sdy {
std::unique_ptr<Pass> createInsertExplicitReshardsPass();
} // namespace sdy
} // namespace mlir

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
        pm.addPass(createClusterDistributedKernelsPass());
      });
}

} // namespace mlir::enzyme::distributed
