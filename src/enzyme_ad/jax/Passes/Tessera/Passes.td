#ifndef ENZYME_AD_JAX_TESSERA_PASSES
#define ENZYME_AD_JAX_TESSERA_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def LiftTesseraAnnotationsPass : Pass<"lift-tessera-annotations", "ModuleOp"> {
  let summary = "Extracts tessera_op and tessera_optimize annotations from LLVM global annotations; adds MLIR tessera_op string attributes and optimization rules";
  let description = [{
    This pass finds tessera_op annotations in the LLVM global annotations and turns them MLIR tessera_op attributes on the appropriate function. It also finds tessera_optimize annotations and collects all of the rule strings into a vector which it stores as a module attribute.

    A function annotated `tessera_no_rewrite` gets a `tessera.no_rewrite` attribute, which keeps tessera-apply-pdl from rewriting anything in its body. The pipeline inlines before this pass runs, so such a function must also be noinline.

    The plugin's `tessera::guarantees`, `tessera::assumes` and `tessera::preserves` attributes arrive as `tessera_guarantees=SPD:return` / `tessera_guarantees=SPD:arg2`, `tessera_assumes=SPD:arg1` and `tessera_preserves=SPD:0,1`. What is always true goes on the definition as `tessera.property = ["SPD"]`: a guarantee on the result, on the sret parameter of a function returning through one, or on the argument its tessera_op marks val=out; an assumption on the parameter. A preserves stays a rule on the function, `tessera.preserves = [...]`, each entry naming a property and the parameters (sret included) it is preserved from; it applies to the return value, or else to the one argument the tessera_op marks val=out or val=inout. tessera-propagate-properties turns these into facts on call arguments.
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

def LowerTesseraGuardsPass : Pass<"tessera-lower-guards", "ModuleOp"> {
  let summary = "Turn each tessera.guard into a runtime check and a branch";
  let description = [{
    This pass synthesizes the code that tests the condition recorded on a
    tessera.guard, then replaces the guard with a conditional branch between its
    two regions: the specialized rewrite when the condition holds, the original
    computation when it does not. It runs before tessera-to-llvm, so both
    branches still contain tessera.call ops.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "enzyme::tessera::TesseraDialect"
  ];
  let options = [
    Option<"maxUnrolledElems", "max-unrolled-elems", "int64_t",
           /*default=*/"64",
           "Largest matrix, in elements, whose property check is emitted as "
           "straight-line comparisons. A predicate declines above this rather "
           "than emitting an unreasonable amount of code.">
  ];
}

def TesseraPropagatePropertiesPass : Pass<"tessera-propagate-properties", "ModuleOp"> {
  let summary = "Record on each tessera.call argument the properties known of the value passed there";
  let description = [{
    Derives facts from the declarations lift-tessera-annotations placed -- a `tessera.property` on a definition's output (guaranteed on exit) or input parameter (assumed on entry), and `tessera.preserves` rules on functions -- and records each fact on the tessera.call arguments the value is passed as, as `tessera.property = ["SPD"]` in the call's arg_attrs. A rule's condition, `if SPD(A), ...`, reads these facts. Facts already recorded are kept, and implications (SPD is symmetric) are applied when a fact is read, not written here.

    Run it after mem2reg, so that a value flows from the call that produced it to the call that uses it rather than through memory.
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
