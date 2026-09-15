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

// Lowering pipeline run on a fully-decided distributed-search candidate
// before scoring. Currently runs InlineDeviceLocalAxes followed by
// LowerKernels; more lowering stages will be added here as they come online
// (see StrategyScorer in SearchStrategies.cpp). `lowerLogicalAxes` should
// normally stay true here:
// candidates reaching this pipeline are expected to already have every
// logical axis decided (e.g. by the search's heuristic completer), so there
// should be no un-sharded logical axis left to preserve.
void buildDistributedSearchLoweringPipeline(OpPassManager &pm,
                                            bool lowerLogicalAxes = true);
void registerDistributedSearchLoweringPipeline();

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
