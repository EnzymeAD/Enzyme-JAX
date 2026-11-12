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

def StripPerfifyPass : Pass<"strip-perfify", "ModuleOp"> {
  let summary = "Erase all Perfify ops so the module can be lowered.";
  let description = [{
    Perfify ops are analysis-only and have no lowering. Run this after the
    analysis input has been printed, before lowering out of structured
    control flow, which would leave loop hypotheses outside any loop.
  }];
}

def PerfifyToLeanPass : Pass<"perfify-to-lean", "ModuleOp"> {
  let summary = "Collect unproven Perfify conditions for export to Lean.";

  let options = [
    Option<"output", "output", "std::string", /*default=*/"\"-\"",
           "Path of the Lean file to write, or '-' for stdout">,
    ListOption<"imports", "imports", "std::string",
               "Modules to import (default: Mathlib, Cslib's TimeM)">
  ];

  let dependentDialects = ["enzyme::perfify::PerfifyDialect"];
}

def LeanToPerfifyPass : Pass<"lean-to-perfify", "ModuleOp"> {
  let summary = "Mark the conditions whose Lean proofs check as proven.";
  let options = [
    Option<"proof", "proof", "std::string", /*default=*/"",
           "Path of the completed Lean file">,
    Option<"lake", "lake", "std::string", /*default=*/"\"lake\"",
           "The lake executable">,
    Option<"project", "project", "std::string", /*default=*/"\".\"",
           "Lake project in which the imports are built">,
    ListOption<"imports", "imports", "std::string",
               "Modules to import, as for perfify-to-lean">
  ];
  let dependentDialects = ["enzyme::perfify::PerfifyDialect"];
}

#endif // ENZYME_AD_JAX_PERFIFY_PASSES
