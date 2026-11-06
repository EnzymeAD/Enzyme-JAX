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
  let summary = "Lower Comm ops to `enzymexla.jit_call`";
  let dependentDialects = [
    "comm::CommDialect",
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
    "enzyme::EnzymeDialect",
    "LLVM::LLVMDialect",
  ];
  let options = [
    Option<"order_jit_calls", "order-jit-calls", "bool", /*default=*/"false",
           "Thread tokens through communication JIT calls to preserve ordering">,
    Option<"mpi_request_width", "mpi_request_width", "unsigned", /*default=*/"64",
           "Bit width of `MPI_Request` in the target MPI implementation "
           "(e.g. 32 for MPICH, 64 for Open MPI on 64-bit platforms)">,
  ];
}

def LegalizeMpiToNcclPass : Pass<"legalize-mpi-to-nccl"> {
  let summary = "Lower comm.mpi ops to comm.nccl ops";
  let dependentDialects = [
    "comm::CommDialect",
    "stablehlo::StablehloDialect",
  ];
}

#endif // ENZYMEXLA_PASSES_COMM_PASSES_TD
