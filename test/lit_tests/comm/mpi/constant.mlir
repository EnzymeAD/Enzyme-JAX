// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main() -> (!comm.mpi_comm, !comm.mpi_op) {
func.func @main() -> (!comm.mpi_comm, !comm.mpi_op) {
    // CHECK: %[[v0:.*]] = comm.mpi.constant #comm.mpi.comm<MPI_COMM_WORLD> : !comm.mpi_comm
    %0 = comm.mpi.constant #comm.mpi_comm<MPI_COMM_WORLD> : !comm.mpi_comm
    // CHECK: %[[v1:.*]] = comm.mpi.constant #comm.mpi.op<MPI_OP_NULL> : !comm.mpi_op
    %1 = comm.mpi.constant #comm.mpi_op<MPI_OP_NULL> : !comm.mpi_op
    return %0, %1 : !comm.mpi_comm, !comm.mpi_op
}
