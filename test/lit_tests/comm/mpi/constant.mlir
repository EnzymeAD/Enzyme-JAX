// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main() -> (!comm.mpi.comm, !comm.mpi.op) {
func.func @main() -> (!comm.mpi.comm, !comm.mpi.op) {
    // CHECK: %[[v0:.*]] = comm.mpi.constant #comm.mpi.comm<MPI_COMM_WORLD> : !comm.mpi.comm
    %0 = comm.mpi.constant #comm.mpi.comm<MPI_COMM_WORLD> : !comm.mpi.comm
    // CHECK: %[[v1:.*]] = comm.mpi.constant #comm.mpi.op<MPI_OP_NULL> : !comm.mpi.op
    %1 = comm.mpi.constant #comm.mpi.op<MPI_OP_NULL> : !comm.mpi.op
    return %0, %1 : !comm.mpi.comm, !comm.mpi.op
}
