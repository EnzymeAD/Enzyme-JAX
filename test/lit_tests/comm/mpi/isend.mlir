// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[TAG:.*]]: tensor<i32>, %[[DEST:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.mpi_comm) -> !comm.mpi_request {
func.func @main(%buf : tensor<4xf32>, %tag : tensor<i32>, %dest : tensor<i32>, %comm : !comm.mpi_comm) -> !comm.mpi_request {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.isend %[[BUF]], %[[TAG]], %[[DEST]], %[[COMM]] : (tensor<4xf32>, tensor<i32>, tensor<i32>, !comm.mpi_comm) -> !comm.mpi_request
    %0 = comm.mpi.isend %buf, %tag, %dest, %comm : (tensor<4xf32>, tensor<i32>, tensor<i32>, !comm.mpi_comm) -> !comm.mpi_request
    return %0 : !comm.mpi_request
}
