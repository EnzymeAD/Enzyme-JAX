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
    "enzyme::distributed::DistributedDialect"
  ];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
