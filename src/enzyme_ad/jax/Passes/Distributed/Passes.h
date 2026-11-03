#ifndef DISTRIBUTED_PASSES_H
#define DISTRIBUTED_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

// Need it within pass declaration
#include <string>

namespace mlir {
namespace enzyme {
namespace distributed {

void registerShardyToDistributedPipeline();

// Explicitly runs the op verifiers of every DistributedKernelOp,
// DistributedFunctionOp and DistributedCallOp under `root`, emitting each
// failing op's own diagnostic. Independent of any PassManager verifier setting.
LogicalResult verifyDistributedOps(Operation *root);

// Lowering pipeline run on a fully-decided distributed-search candidate,
// either for scoring (a disposable clone, see cloneAndApplyDecisions in
// SearchStrategies.cpp) or as the real final lowering of the search's
// winner. It takes the candidate from physical-axis sharding down through
// per-kernel local lowering (RefinePartitioningSlots,
// CanonicalizeShardedFactorOrder, InlineDeviceLocalAxes, LowerKernels),
// kernel-body cleanup and collective fusion/atomization, and
// AnnotateKernelLatency.
//
// `lowerLogicalAxes` should normally stay true: candidates reaching this
// pipeline are expected to already have every logical axis decided (e.g. by
// the search's heuristic completer), so there should be no un-sharded
// logical axis left to preserve. `dumpKernelModulesTo` is forwarded to
// AnnotateKernelLatency.
//
// distributed-inline-kernels and distributed-lower-collectives always run
// after AnnotateKernelLatency, materializing the kernel-dissolved,
// collective-lowered IR shape that ScheduleGraphAnalysis needs to build a
// correct event graph from, for every candidate reaching schedule-graph-
// building, not only the winner (see cloneAndApplyDecisions in
// SearchStrategies.cpp for the scoring clone's own use of this pipeline).
// `finalizeSchedule` gates only the last of the three, distributed-reorder-
// schedule, which physically reorders ops to match the simulated schedule
// and is meaningful just once, on the single candidate the search ultimately
// picks; it defaults to true for the real pipeline, but per-candidate
// scoring passes false explicitly.
//
// `reorderForLogging` also runs distributed-reorder-schedule when
// `finalizeSchedule` is false, so the 03-final logging checkpoint shows
// post-reorder IR for a non-winning candidate too. cloneAndApplyDecisions
// passes true exactly when logging is active, for every candidate.
void buildDistributedSearchLoweringPipeline(
    OpPassManager &pm, bool lowerLogicalAxes = true,
    llvm::StringRef dumpKernelModulesTo = "", bool finalizeSchedule = true,
    bool reorderForLogging = false);
void registerDistributedSearchLoweringPipeline();

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
