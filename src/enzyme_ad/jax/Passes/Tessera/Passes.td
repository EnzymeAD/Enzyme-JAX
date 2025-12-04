#ifndef ENZYME_AD_JAX_TESSERA_PASSES
#define ENZYME_AD_JAX_TESSERA_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def FuncToTesseraPass : Pass<"func-to-tessera"> {
  let summary = "Convert operations in the Func Dialect to operations in the Tessera Dialect";
  let description = [{
    This pass checks if an operation is marked with a custom
    annotation and if so, creates a TesseraOp from the FuncOp.
  }];
  let constructor = "mlir::enzyme::tessera::createFuncToTesseraPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

def TesseraToFuncPass : Pass<"tessera-to-func"> {
  let summary = "Convert operations in the Tessera Dialect to operations in the Func Dialect";
  let description = [{
    This pass converts a TesseraOp back into a FuncOp.
  }];
  let constructor = "mlir::enzyme::tessera::createTesseraToFuncPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "enzyme::tessera::TesseraDialect"
  ];
}

#endif // ENZYME_AD_JAX_TESSERA_PASSES