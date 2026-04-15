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

def InsertIdentityReshardPass
    : Pass<"insert-identity-reshard", "func::FuncOp"> {
  let summary = "Insert identity reshard ops on tensor operands";
  let description = [{
    This prepass inserts an identity sdy.reshard on every tensor operand so
    the distributed lowering sees a graph where tensor values are explicitly
    separated from the communication boundaries that define their sharding.
  }];
  let dependentDialects = [
    "sdy::SdyDialect",
    "func::FuncDialect",
  ];
}

// TODO(distributed-ir-migration): Re-enable these pass defs one-by-one as
// each pass implementation is migrated to the unified MeshComputation IR.
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

def ParameterizedLocalizeDistributedPass
    : Pass<"parameterized-localize-distributed",
           "distributed::MeshComputationOp"> {
  let summary = "Prototype parameterized localization for one mesh computation";
  let description = [{
    Prototype pass that localizes a single MPMD axis within one
    distributed.MeshComputationOp.
    The first implementation only resolves the target axis binding and
    validates the requested tensor axis; communication and scheduling rewrites
    are deferred.
  }];
  let options = [
    Option<
      /*C++ variable name=*/"targetMpmdAxisIndex",
      /*CLI argument=*/"target-mpmd-axis-index",
      /*type=*/"int64_t",
      /*default=*/"-1",
      /*description=*/"Index of the MPMD axis to localize">,
    Option<
      /*C++ variable name=*/"tensorAxis",
      /*CLI argument=*/"tensor-axis",
      /*type=*/"int64_t",
      /*default=*/"0",
      /*description=*/"Tensor axis to bind the target MPMD axis to">,
  ];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

def LocalizeDistributedModulePass : Pass<"localize-distributed-module", "ModuleOp"> {
  let summary = "Run deterministic pseudo-random localization on all distributed mesh computations in a module";
  let description = [{
    This wrapper pass finds every distributed.MeshComputationOp in the module,
    generates a fixed-seed pseudo-random localization configuration for it, and
    applies the parameterized localization entrypoint. The generated choices are
    intentionally a stand-in for the future search procedure.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
    "sdy::SdyDialect",
  ];
}

def PrintTimingAnalysisPass
    : Pass<"print-timing-analysis", "distributed::MeshComputationOp"> {
  let summary = "Print timing analysis results for each lane and op";
  let description = [{
    This debugging pass computes distributed timing analysis for a
    MeshComputationOp and prints the resulting time range for each op in each
    lane.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
  ];
}

def PrintTimingAnalysisModulePass
    : Pass<"print-timing-analysis-module", "ModuleOp"> {
  let summary = "Print timing analysis results for all distributed mesh computations";
  let description = [{
    This wrapper pass finds every distributed.MeshComputationOp in the
    module and prints timing analysis results for each lane and op.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

def DistributedSimplifyCollectivesPass
    : Pass<"distributed-simplify-collectives", "func::FuncOp"> {
  let summary = "Simplify distributed collective operations";
  let description = [{
    This pass simplifies collective operations within functions that contain
    MeshComputationOps. An identity collective is one where:
    (1) input and output shardings are equal,
    (2) input and output tensor types are equal, and
    (3) in every device body, both Send and Recv for the token coexist
        (i.e. the data stays on the same device).
    For such collectives, the Send/Recv pairs are replaced with direct SSA
    forwarding, and the Collective and its Transfer op are erased.
  }];
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect",
    "func::FuncDialect",
  ];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
