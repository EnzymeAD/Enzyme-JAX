#ifndef ENZYMEXLA_COMM_PASSES
#define ENZYMEXLA_COMM_PASSES

include "mlir/Pass/PassBase.td"

def CommFlattenSplitsPass : Pass<"comm-flatten-splits"> {
  let summary = "Splits branches and specializes subsplits of a top-level CommSplit until it contains no subsplits.";
  let dependentDialects = ["comm"];
}

#endif // ENZYMEXLA_COMM_PASSES