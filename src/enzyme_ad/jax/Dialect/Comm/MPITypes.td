#ifndef ENZYMEXLA_COMM_DIALECT_MPI_TYPES_TD
#define ENZYMEXLA_COMM_DIALECT_MPI_TYPES_TD

include "Dialect.td"

include "mlir/IR/CommonTypeConstraints.td"

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

#endif // ENZYMEXLA_COMM_DIALECT_MPI_TYPES_TD
