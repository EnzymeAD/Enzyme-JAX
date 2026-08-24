#ifndef ENZYMEXLA_COMM_DIALECT_TYPES_TD
#define ENZYMEXLA_COMM_DIALECT_TYPES_TD

include "Dialect.td"

include "mlir/IR/CommonTypeConstraints.td"

// MPI
def MPI_Comm : Comm_Type<"MPI_Comm", "mpi.comm"> {
  let summary = "MPI communicator handler";
}

def MPI_Request : Comm_Type<"MPI_Request", "mpi.request"> {
  let summary = "MPI asynchronous request handler";
}

def MPI_Datatype : Comm_Type<"MPI_Datatype", "mpi.datatype"> {
  let summary = "MPI datatype handler";
}

def MPI_Op : Comm_Type<"MPI_Op", "mpi.op"> {
  let summary = "MPI reduction operator handler";
}

// NCCL
def NCCL_Comm : Comm_Type<"NCCL_Comm", "nccl.comm"> {
  let summary = "NCCL communicator handler";
}

def NCCL_Result : Comm_Type<"NCCL_Result", "nccl.result"> {
  let summary = "NCCL result handler";
}

def NCCL_Stream : Comm_Type<"NCCL_Stream", "nccl.stream"> {
  let summary = "NCCL CUDA stream handler";
}

def NCCL_RedOp : Comm_Type<"NCCL_RedOp", "nccl.redop"> {
  let summary = "NCCL reduction operation handler";
}

#endif // ENZYMEXLA_COMM_DIALECT_TYPES_TD
