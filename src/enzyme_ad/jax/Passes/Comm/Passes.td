#ifndef ENZYMEXLA_PASSES_COMM_PASSES_TD
#define ENZYMEXLA_PASSES_COMM_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerCommToStableHLOPass : Pass<"lower-comm-to-stablehlo"> {
  let summary = "Lower Comm ops to `stablehlo.custom_call`";
  let dependentDialects = [
    "comm::CommDialect",
    "stablehlo::StablehloDialect",
  ];
}

#endif // ENZYMEXLA_PASSES_COMM_PASSES_TD
