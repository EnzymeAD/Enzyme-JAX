#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::enzyme::distributed {

void buildDistributedSearchLoweringPipeline(OpPassManager &pm,
                                            bool lowerLogicalAxes) {
  LowerKernelsPassOptions options;
  options.lowerLogicalAxes = lowerLogicalAxes;
  pm.addPass(createLowerKernelsPass(options));
}

namespace {
// Pipeline-level mirror of LowerKernelsPass's lowerLogicalAxes option, for
// CLI use (see buildDistributedSearchLoweringPipeline for the option's
// meaning in this pipeline's context).
struct DistributedSearchLoweringPipelineOptions
    : public PassPipelineOptions<DistributedSearchLoweringPipelineOptions> {
  Option<bool> lowerLogicalAxes{
      *this, "lower-logical-axes",
      llvm::cl::desc(
          "Whether to lower sharding over logical axes. Should normally "
          "stay true: candidates reaching this pipeline are expected to "
          "already have every logical axis decided. Default: true"),
      llvm::cl::init(true)};
};
} // namespace

void registerDistributedSearchLoweringPipeline() {
  PassPipelineRegistration<DistributedSearchLoweringPipelineOptions>(
      "distributed-search-lowering-pipeline",
      "Lowering pipeline run on a fully-decided distributed-search "
      "candidate before scoring. Currently just runs LowerKernels; more "
      "lowering stages will be added here as they come online.",
      [](OpPassManager &pm,
         const DistributedSearchLoweringPipelineOptions &options) {
        buildDistributedSearchLoweringPipeline(pm, options.lowerLogicalAxes);
      });
}

} // namespace mlir::enzyme::distributed
