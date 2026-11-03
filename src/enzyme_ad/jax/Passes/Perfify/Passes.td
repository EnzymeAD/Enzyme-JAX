#ifndef ENZYME_AD_JAX_PERFIFY_PASSES
#define ENZYME_AD_JAX_PERFIFY_PASSES
include "mlir/Pass/PassBase.td"

def SimpleCycleAnalysisPass : Pass<"simple-cycle-analysis", "ModuleOp"> {
  let summary = "Using cost definitions, calculate and check the cost analysis";
  
  let dependentDialects = ["enzyme::perfify::PerfifyDialect"];
}

def LiftPerfifyAnnotationsPass : Pass<"lift-perfify-annotations", "ModuleOp"> {
  let summary = "Lift annotations into Perfify-style Hoare triples.";
  
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "arith::ArithDialect",
    "enzyme::perfify::PerfifyDialect"
  ];
}

def PerfifyToLeanPass : Pass<"perfify-to-lean", "ModuleOp"> {
  let summary = "Collect unproven Perfify conditions for export to Lean.";

  let options = [
    Option<"output", "output", "std::string", /*default=*/"\"-\"",
           "Path of the Lean file to write, or '-' for stdout">
  ];

  let dependentDialects = ["enzyme::perfify::PerfifyDialect"];
}

#endif // ENZYME_AD_JAX_PERFIFY_PASSES
