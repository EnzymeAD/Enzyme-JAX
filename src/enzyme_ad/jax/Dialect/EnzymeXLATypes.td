#ifndef ENZYMEXLA_TYPES
#define ENZYMEXLA_TYPES

include "Enzyme/MLIR/Dialect/Dialect.td"
include "Dialect.td"
include "EnzymeXLAAttrs.td"

def MPI_Comm : EnzymeXLA_Type<"Comm", "comm"> {
  let summary = "MPI communicator handler";
  let description = [{
    This type represents a handler for the MPI communicator.
  }];
}

def MPI_Retval : EnzymeXLA_Type<"Retval", "retval"> {
  let summary = "MPI function call return value (!mpi.retval)";
  let description = [{
    This type represents a return value from an MPI function call.
    This value can be MPI_SUCCESS, MPI_ERR_IN_STATUS, or any error code.

    This return value can be compared agains the known MPI error classes
    represented by `#mpi.errclass` using the `mpi.retval_check` operation.
  }];
}

#endif // ENZYMEXLA_TYPES
