// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: #world = #comm.mpi.comm<MPI_COMM_WORLD>
#world = #comm.mpi.comm<MPI_COMM_WORLD>

// CHECK: #null = #comm.mpi.op<MPI_OP_NULL>
#null = #comm.mpi.op<MPI_OP_NULL>

// CHECK: #self = #comm.mpi.comm<MPI_COMM_SELF>
#self = #comm.mpi.comm<MPI_COMM_SELF>
