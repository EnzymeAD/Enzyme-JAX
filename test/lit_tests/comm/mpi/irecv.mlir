// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[TAG:.*]]: tensor<i32>, %[[SOURCE:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.mpi_comm) -> (tensor<4xf32>, !comm.mpi_request) {
func.func @main(%tag : tensor<i32>, %source : tensor<i32>, %comm : !comm.mpi_comm) -> (tensor<4xf32>, !comm.mpi_request) {
    // CHECK-NEXT: %[[buf:.*]], %[[request:.*]] = comm.mpi.irecv %[[TAG]], %[[SOURCE]], %[[COMM]] : (tensor<i32>, tensor<i32>, !comm.mpi_comm) -> (tensor<4xf32>, !comm.mpi_request)
    %0:2 = comm.mpi.irecv %tag, %source, %comm : (tensor<i32>, tensor<i32>, !comm.mpi_comm) -> (tensor<4xf32>, !comm.mpi_request)
    return %0#0, %0#1 : tensor<4xf32>, !comm.mpi_request
}
