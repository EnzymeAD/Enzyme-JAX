#ifndef ENZYME_AD_JAX_TESSERA_PASSES
#define ENZYME_AD_JAX_TESSERA_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def LiftTesseraAnnotationsPass : Pass<"lift-tessera-annotations", "ModuleOp"> {
  let summary = "Extracts tessera_op and tessera_optimize annotations from LLVM global annotations; adds MLIR tessera_op string attributes and optimization rules";
  let description = [{
    This pass finds tessera_op annotations in the LLVM global annotations and turns them MLIR tessera_op attributes on the appropriate function. It also finds tessera_optimize annotations and collects all of the rule strings into a vector which it stores as a module attribute.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def FuncAttrToTesseraAttrPass : Pass<"func-attr-to-tessera-attr", "ModuleOp"> {
  let summary = "Raise generic tessera_op attribute to custom tessera.convert attribute on operations";
  let description = [{
    This pass checks if a function has a MLIR tessera_op string attribute attached to its 
    declaration and replaces it with the custom tessera.convert attribute.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def ParseOptimizationRulesPass : Pass<"parse-optimization-rules", "ModuleOp"> {
  let summary = "Parses the optimization rule strings and converts them into PDL patterns";
  let description = [{
    This pass finds the tessera optimization rules in the module and parses each of them, forming an AST, then walks the AST and creating PDL operations.
  }];
  let dependentDialects = [
    "enzyme::tessera::TesseraDialect",
    "pdl::PDLDialect"
  ];
}

def LLVMToTesseraPass : Pass<"llvm-to-tessera", "ModuleOp"> {
  let summary = "Convert operations in the LLVM Dialect to operations in the Tessera Dialect";
  let description = [{
    This pass checks if an operation is marked with a custom
    annotation and if so, creates a TesseraOp from the LLVMFuncOp.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def TesseraToLLVMPass : Pass<"tessera-to-llvm", "ModuleOp"> {
  let summary = "Convert operations in the Tessera Dialect to operations in the LLVM Dialect";
  let description = [{
    This pass converts a TesseraOp back into a LLVMOp.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def TesseraApplyPDLPass : Pass<"tessera-apply-pdl", "ModuleOp"> {
  let summary = "Apply PDL patterns to Tessera operations";
  let description = [{
    This pass applies PDL patterns to operations in the Tessera dialect.
  }];
  let dependentDialects = [
    "pdl::PDLDialect",
    "pdl_interp::PDLInterpDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

#endif // ENZYME_AD_JAX_TESSERA_PASSES
