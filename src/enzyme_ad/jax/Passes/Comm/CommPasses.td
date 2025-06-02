#ifndef ENZYMEXLA_COMM_PASSES
#define ENZYMEXLA_COMM_PASSES

include "mlir/Pass/PassBase.td"

def CommDCE : Pass<"comm-dce"> {
  let summary = "An implementation of the classic DCE pass that is aware of communication edges";
  let dependentDialects = [
    "comm::CommDialect"
  ];
}




#endif // ENZYMEXLA_COMM_PASSES