#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::enzyme::distributed {

void buildDistributedSearchLoweringPipeline(OpPassManager &pm,
                                            bool lowerLogicalAxes) {
  LowerKernelsPassOptions options;
  options.lowerLogicalAxes = lowerLogicalAxes;
  // CanonicalizeShardedFactorOrder must run first: both InlineDeviceLocalAxes
  // and LowerKernels assume every DeviceLocalAxis is already contiguous and
  // minor-most on its dimension, which is exactly the invariant this
  // establishes.
  pm.addPass(createCanonicalizeShardedFactorOrderPass());
  pm.addPass(createInlineDeviceLocalAxesPass());
  pm.addPass(createLowerKernelsPass(options));
  // Can only drop anchors after inlining tensor types
  pm.addPass(createDropIdentityCollectivesPass());
  pm.addPass(createDropIdentityPartitioningAnchorsPass());
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
      "Lowers a fully-decided module written in terms of physical execution "
      "axes to its sharded form.",
      [](OpPassManager &pm,
         const DistributedSearchLoweringPipelineOptions &options) {
        buildDistributedSearchLoweringPipeline(pm, options.lowerLogicalAxes);
      });
}

} // namespace mlir::enzyme::distributed
