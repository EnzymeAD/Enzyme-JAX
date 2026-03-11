#ifndef ENZYME_AD_JAX_DISTRIBUTED_PASSES
#define ENZYME_AD_JAX_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"

def ShardyToDistributedPass : Pass<"shardy-to-distributed", "ModuleOp"> {
  let summary = "Convert Shardy dialect to Distributed dialect";
  let description = [{
    This pass converts operations in the Shardy Dialect to operations in the Distributed Dialect. It 
    does not yet perform any optimizations or localizations. Its main purpose is to reconcile the shardy
    mesh with the distributed mesh representation, including finding the physical mesh.
  }];
  let dependentDialects = [
    "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

def ShardyFunctionToDistributedPass
    : Pass<"shardy-function-to-distributed", "func::FuncOp"> {
  let summary = "Convert one shardy function to Distributed dialect";
  let description = [{
    This pass converts a single function from shardy annotations/ops to the
    Distributed dialect using a selected physical mesh as context.
  }];
  let options = [
    Option<
      /*C++ variable name=*/"physicalMeshSymName",
      /*CLI argument=*/"physical-mesh-sym-name",
      /*type=*/"std::string",
      /*default=*/"\"\"",
      /*description=*/"Symbol name of the top-level distributed.PhysicalMeshOp used for conversion context">
  ];
  let dependentDialects = [
    "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

def ShardyCollectivesToDistributedPass
    : Pass<"shardy-collectives-to-distributed", "func::FuncOp"> {
  let summary = "Rewrite Shardy collectives to Distributed collectives";
  let description = [{
    This pass rewrites Shardy collective ops in a function to the Distributed
    collective/send/recv form, intended to run after shardy-function-to-distributed.
  }];
  let dependentDialects = [
    "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

def LocalizeDistributedPass : Pass<"localize-distributed", "distributed::MeshComputationOp"> {
  let summary = "Based on a localization and partitioning strategy, localize a mesh computation op to a region computation.";
  let description = [{
    This pass takes a mesh computation op and localizes it to a region computation op. For each region 
    suggested by the partitioning strategy, a new region computation op is created. The region op uses
    a submesh suggested by the localization strategy. Ops formerly within the mesh computation are de-sharded
    and each chunk is placed in the appropriate region. }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
    "sdy::SdyDialect", // still dependent on their sharding attributes.
  ];
}

def LocalizeDistributedModulePass : Pass<"localize-distributed-module", "ModuleOp"> {
  let summary = "Run localize-distributed on all distributed mesh computations in a module";
  let description = [{
    This wrapper pass finds distributed.MeshComputation ops inside the module
    and runs localize-distributed on each one.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
    "sdy::SdyDialect",
  ];
}

def PrintTimingAnalysisPass
    : Pass<"print-timing-analysis", "distributed::RegionComputationOp"> {
  let summary = "Print timing analysis results for each lane and op";
  let description = [{
    This debugging pass computes the distributed timing analysis for a
    RegionComputationOp and prints the resulting time range for each op in each
    lane.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
  ];
}

def PrintTimingAnalysisModulePass
    : Pass<"print-timing-analysis-module", "ModuleOp"> {
  let summary = "Print timing analysis results for all distributed regions";
  let description = [{
    This wrapper pass finds every distributed.RegionComputationOp in the
    module and prints timing analysis results for each lane and op.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
