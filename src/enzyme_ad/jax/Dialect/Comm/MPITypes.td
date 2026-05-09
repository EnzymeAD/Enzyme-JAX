#ifndef ENZYMEXLA_COMM_DIALECT_MPI_TYPES_TD
#define ENZYMEXLA_COMM_DIALECT_MPI_TYPES_TD

include "Dialect.td"

def MPI_Comm : Comm_Type<"MPI_Comm", "mpi.comm"> {
  let summary = "MPI communicator handler";
  let description = [{
    This type represents a handler for a MPI communicator.
  }];
}

def MPI_Request : Comm_Type<"MPI_Request", "mpi.request"> {
  let summary = "MPI asynchronous request handler";
  let description = [{
    This type represents a handler to an asynchronous request.
  }];
}

#endif // ENZYMEXLA_COMM_DIALECT_MPI_TYPES_TD
