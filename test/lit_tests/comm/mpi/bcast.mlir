// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[ROOT:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.mpi_comm)  -> tensor<4xf32> {
func.func @main(%buf : tensor<4xf32>, %root : tensor<i32>, %comm : !comm.mpi_comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.bcast %[[BUF]], %[[ROOT]], %[[COMM]] : (tensor<4xf32>, tensor<i32>, !comm.mpi_comm) -> tensor<4xf32>
    %0 = comm.mpi.bcast %buf, %root, %comm : (tensor<4xf32>, tensor<i32>, !comm.mpi_comm) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
