#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::enzyme::distributed {

void buildDistributedSearchLoweringPipeline(
    OpPassManager &pm, bool lowerLogicalAxes,
    llvm::StringRef dumpKernelModulesTo, bool finalizeSchedule,
    llvm::StringRef actTarget, llvm::StringRef actWorkerDirRoot,
    llvm::StringRef actWorkerScriptDir) {
  LowerKernelsPassOptions options;
  options.lowerLogicalAxes = lowerLogicalAxes;
  // CanonicalizeShardedFactorOrder must run first: both InlineDeviceLocalAxes
  // and LowerKernels assume every DeviceLocalAxis is already contiguous and
  // minor-most on its dimension, which is exactly the invariant this
  // establishes.
  pm.addPass(createRefinePartitioningSlotsPass());
  pm.addPass(createCanonicalizeShardedFactorOrderPass());
  pm.addPass(createInlineDeviceLocalAxesPass());
  // InlineDeviceLocalAxes builds its axis-algebra ops (axis.getaxis,
  // axis.factor, axis.product, axis.map) at each rewrite site rather than a
  // shared location, so the same computation is often rebuilt at several
  // nearby sites; those ops are Pure, so CSE collapses the duplicates.
  pm.addPass(createCSEPass());
  GreedyRewriteConfig canonConfig;
  canonConfig.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  pm.addPass(createCanonicalizerPass(canonConfig));
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
  // These two run here, after the kernel-side cleanup above and before
  // LowerKernelsToExecutable: MakeReplicationsExplicit needs a module whose
  // collectives' mesh bookkeeping is otherwise settled (no more identity
  // collectives to fold away, no dangling sharding metadata) before it
  // commits to explicit replication factors, and AtomizeCollectives'
  // idempotent per-atom rewrite is cheapest to run once that's the only
  // thing left changing collective operands. LowerKernelsToExecutable wants
  // the fully merged, cleaned-up state that follows.
  pm.addPass(createMakeReplicationsExplicitPass());
  // Fusion needs explicit replication (every mesh atom on both sides of both
  // collectives) but resolves its own atom space, so it does not need atomic
  // input; atomize then runs last over the fused and unfused collectives
  // alike, and is a no-op on the atomic collectives fusion builds.
  pm.addPass(createFuseCollectivesPass());
  pm.addPass(createAtomizeCollectivesPass());
  // Runs after every kernel-merging/cleanup pass above, so each kernel is
  // hoisted exactly once, over its final body, rather than possibly once per
  // merged constituent. Must precede LowerKernelsToExecutable
  // unconditionally, like that pass, regardless of finalizeSchedule: ACT's
  // cost model has no notion of stablehlo.partition_id, so every kernel
  // reaching ACT dispatch, whether the eventual winner or a per-candidate
  // scoring clone, needs it hoisted out first.
  pm.addPass(createHoistKernelPartitionIdPass());
  // Runs last so that whatever it dispatches (or, for now, dumps) per kernel
  // reflects the fully merged and cleaned-up IR above, not an intermediate
  // state with dead axis-algebra ops still attached.
  LowerKernelsToExecutablePassOptions executableOptions;
  executableOptions.dumpKernelModulesTo = dumpKernelModulesTo.str();
  executableOptions.actTarget = actTarget.str();
  executableOptions.actWorkerDirRoot = actWorkerDirRoot.str();
  executableOptions.actWorkerScriptDir = actWorkerScriptDir.str();
  pm.addPass(createLowerKernelsToExecutablePass(executableOptions));

  // Kernel inlining and collective lowering materialize real physical IR
  // (spliced-in kernel bodies, StableHLO async op chains for collectives),
  // but neither depends on which candidate wins the search: every
  // candidate's ScheduleGraphAnalysis needs to see the same fully
  // materialized shape, kernel wrappers dissolved and collectives already
  // lowered, that scoring and finalization both cost from, so both always
  // run here rather than being gated to the winner. Schedule reordering is
  // different: it physically reorders ops to match a simulated admission
  // order, which is only meaningful, and only wanted, once on the single
  // candidate ultimately chosen, so it alone stays behind finalizeSchedule.
  pm.addPass(createInlineKernelsPass());
  pm.addPass(createLowerCollectivesPass());
  if (finalizeSchedule)
    pm.addPass(createReorderSchedulePass());
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
  Option<std::string> dumpKernelModulesTo{
      *this, "dump-kernel-modules-to",
      llvm::cl::desc("Forwarded to distributed-lower-kernels-to-executable."),
      llvm::cl::init("")};
  Option<bool> finalizeSchedule{
      *this, "finalize-schedule",
      llvm::cl::desc(
          "Whether to run distributed-reorder-schedule, physically "
          "reordering ops to match the simulated schedule, after lowering. "
          "distributed-inline-kernels and distributed-lower-collectives "
          "always run regardless of this option. Should normally stay "
          "true; set to false to inspect the pipeline's output before "
          "reordering. Default: true"),
      llvm::cl::init(true)};
  Option<std::string> actTarget{
      *this, "act-target",
      llvm::cl::desc(
          "Forwarded to distributed-lower-kernels-to-executable's option of "
          "the same name; see there for what activates real ACT dispatch."),
      llvm::cl::init("")};
  Option<std::string> actWorkerDirRoot{
      *this, "act-worker-dir-root",
      llvm::cl::desc("Forwarded to distributed-lower-kernels-to-executable."),
      llvm::cl::init("")};
  Option<std::string> actWorkerScriptDir{
      *this, "act-worker-script-dir",
      llvm::cl::desc("Forwarded to distributed-lower-kernels-to-executable."),
      llvm::cl::init("")};
};
} // namespace

void registerDistributedSearchLoweringPipeline() {
  PassPipelineRegistration<DistributedSearchLoweringPipelineOptions>(
      "distributed-search-lowering-pipeline",
      "Lowers a fully-decided module written in terms of physical execution "
      "axes to its sharded form.",
      [](OpPassManager &pm,
         const DistributedSearchLoweringPipelineOptions &options) {
        buildDistributedSearchLoweringPipeline(
            pm, options.lowerLogicalAxes, options.dumpKernelModulesTo,
            options.finalizeSchedule, options.actTarget,
            options.actWorkerDirRoot, options.actWorkerScriptDir);
      });
}

} // namespace mlir::enzyme::distributed
