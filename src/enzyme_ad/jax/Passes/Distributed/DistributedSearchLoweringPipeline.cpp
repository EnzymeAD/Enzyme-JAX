#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

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
  // InlineDeviceLocalAxes builds its axis-algebra ops (axis.getaxis,
  // axis.factor, axis.product, axis.map) at each rewrite site rather than a
  // shared location, so the same computation is often rebuilt at several
  // nearby sites; those ops are Pure, so CSE collapses the duplicates.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerKernelsPass(options));
  // distributed-lower-kernels is the last real consumer of kernel body ops'
  // own distributed.argument_shardings/output_shardings and of Shardy's
  // sharding-rule attributes; once it's done they're stale (and the former
  // can go dangling if a later pass, e.g. merge-adjacent-trivial-kernels,
  // shrinks or empties partitioning_axes).
  pm.addPass(createDropKernelBodyShardingAttrsPass());
  pm.addPass(createDropShardingRuleAttrsPass());
  // Every trivial kernel (not just ones that happen to be merge-adjacent)
  // should have its own sharding metadata blanked too, so the axis-algebra
  // chain feeding its now-unused partitioning_axes can be swept up below.
  pm.addPass(createDropTrivialKernelShardingPass());
  // Can only drop anchors after inlining tensor types
  pm.addPass(createDropIdentityCollectivesPass());
  pm.addPass(createDropIdentityPartitioningAnchorsPass());
  // Must run after the anchor/cast folds above: a trivial kernel's boundary
  // casts need to already be gone for its operand/result types to line up
  // with its body's block-args/yield, which is what marks it mergeable.
  pm.addPass(createMergeAdjacentTrivialKernelsPass());
  // The passes above can each orphan more axis-algebra ops (blanked
  // partitioning_axes, folded collectives/anchors, merged kernel operands),
  // with nothing after them to clean up until now.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  // Runs last so that whatever it dispatches (or, for now, dumps) per kernel
  // reflects the fully merged and cleaned-up IR above, not an intermediate
  // state with dead axis-algebra ops still attached.
  pm.addPass(createLowerKernelsToExecutablePass());
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
