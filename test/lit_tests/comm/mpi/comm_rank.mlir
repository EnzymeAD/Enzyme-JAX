// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi_comm) -> tensor<i32> {
func.func @main(%comm : !comm.mpi_comm) -> tensor<i32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.comm_rank %[[COMM]] : tensor<i32>
    %0 = comm.mpi.comm_rank %comm : tensor<i32>
    return %0 : tensor<i32>
}
