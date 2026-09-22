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

// Lowering pipeline run on a fully-decided distributed-search candidate,
// either for scoring (a disposable clone, see cloneAndApplyDecisions in
// SearchStrategies.cpp) or as the real final lowering of the search's
// winner. It takes the candidate from physical-axis sharding down through
// per-kernel local lowering (RefinePartitioningSlots,
// CanonicalizeShardedFactorOrder, InlineDeviceLocalAxes, LowerKernels),
// kernel-body cleanup and collective fusion/atomization, and
// LowerKernelsToExecutable.
//
// `lowerLogicalAxes` should normally stay true: candidates reaching this
// pipeline are expected to already have every logical axis decided (e.g. by
// the search's heuristic completer), so there should be no un-sharded
// logical axis left to preserve. `dumpKernelModulesTo` is forwarded to
// LowerKernelsToExecutable.
//
// `finalizeSchedule` gates a further, winner-only finalization stage
// (distributed-inline-kernels, distributed-lower-collectives,
// distributed-reorder-schedule) that materializes real physical IR:
//   - inlined kernel bodies
//   - StableHLO async op chains for collectives
//   - ops physically reordered to match the simulated schedule
// It defaults to true for the real pipeline, but per-candidate scoring must
// stay a fast, read-only simulation and passes false explicitly (see
// cloneAndApplyDecisions).
void buildDistributedSearchLoweringPipeline(
    OpPassManager &pm, bool lowerLogicalAxes = true,
    llvm::StringRef dumpKernelModulesTo = "", bool finalizeSchedule = true);
void registerDistributedSearchLoweringPipeline();

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
