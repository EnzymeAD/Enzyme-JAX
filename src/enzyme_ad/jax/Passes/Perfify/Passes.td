#ifndef ENZYME_AD_JAX_PERFIFY_PASSES
#define ENZYME_AD_JAX_PERFIFY_PASSES
include "mlir/Pass/PassBase.td"

def SimpleCycleAnalysisPass : Pass<"simple-cycle-analysis", "ModuleOp"> {
  let summary = "Using cost definitions, calculate and check the cost analysis";
  
  let dependentDialects = ["enzyme::perfify::PerfifyDialect"];
}


#endif // ENZYME_AD_JAX_PERFIFY_PASSES