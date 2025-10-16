#ifndef ENZYME_AD_JAX_PASSES_TESSERA_PASSES_TD
#define ENZYME_AD_JAX_PASSES_TESSERA_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuncToTesseraPass : Pass<"func-to-tessera"> {
  let summary = "Convert operations in the FuncDialect to operations in the TesseraDialect and vice versa";
  let dependentDialects = [
    "func::FuncDialect",
    "tessera::TesseraDialect"
  ];
}

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PASSES_TD