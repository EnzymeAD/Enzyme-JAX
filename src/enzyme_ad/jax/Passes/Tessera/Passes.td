#ifndef ENZYME_AD_JAX_TESSERA_PASSES
#define ENZYME_AD_JAX_TESSERA_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def TesseraAnnotationToAttributePass : Pass<"tessera-annotation-to-attribute", "ModuleOp"> {
  let summary = "Extracts tessera_op annotations from LLVM global annotations and converts them into MLIR tessera_op string attributes";
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}

def FuncAttrToTesseraAttrPass : Pass<"func-attr-to-tessera-attr", "ModuleOp"> {
  let summary = "Raise generic tessera_op func attribute to custom tessera.convert attribute on operations";
  let description = [{
    This pass checks if a function has a MLIR tessera_op string attribute attached to its 
    declaration and replaces it with the custom tessera.convert attribute.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def FuncToTesseraPass : Pass<"func-to-tessera", "ModuleOp"> {
  let summary = "Convert operations in the Func Dialect to operations in the Tessera Dialect";
  let description = [{
    This pass checks if an operation is marked with a custom
    annotation and if so, creates a TesseraOp from the FuncOp.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def TesseraToFuncPass : Pass<"tessera-to-func", "ModuleOp"> {
  let summary = "Convert operations in the Tessera Dialect to operations in the Func Dialect";
  let description = [{
    This pass converts a TesseraOp back into a FuncOp.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

#endif // ENZYME_AD_JAX_TESSERA_PASSES
