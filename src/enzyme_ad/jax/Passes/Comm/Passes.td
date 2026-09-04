#ifndef ENZYMEXLA_PASSES_COMM_PASSES_TD
#define ENZYMEXLA_PASSES_COMM_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerCommToStablehloPass : Pass<"lower-comm-to-stablehlo"> {
  let summary = "Lower Comm ops to `stablehlo.custom_call`";
  let dependentDialects = [
    "comm::CommDialect",
    "stablehlo::StablehloDialect",
  ];
}

def LowerCommToJITPass : Pass<"lower-comm-to-jit"> {
  let summary = "Lower Comm ops to `stablehlo.jit_call`";
  let dependentDialects = [
    "comm::CommDialect",
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
    "LLVM::LLVMDialect",
  ];
}

#endif // ENZYMEXLA_PASSES_COMM_PASSES_TD
