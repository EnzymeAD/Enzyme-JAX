// RUN: enzymexlamlir-opt %s | FileCheck %s


#world = #comm.mpi.comm<MPI_COMM_WORLD>
#null = #comm.mpi.comm<"MPI_COMM_NULL">
#self = #comm.mpi.comm<MPI_COMM_SELF>

// XFAIL: *
#error = #comm.mpi.comm<MPI_COMM_ERROR>
