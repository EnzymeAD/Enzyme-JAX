// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[TAG:.*]]: i32, %[[SOURCE:.*]]: i32, %[[COMM:.*]]: !comm.mpi.comm) -> (tensor<4xf32>, !comm.mpi.request) {
func.func @main(%tag : i32, %source : i32, %comm : !comm.mpi.comm) -> (tensor<4xf32>, !comm.mpi.request) {
    // CHECK-NEXT: %[[buf:.*]], %[[request:.*]] = comm.mpi.irecv %[[TAG]], %[[SOURCE]], %[[COMM]] : tensor<4xf32>, !comm.mpi.request
    %0:2 = comm.mpi.irecv %tag, %source, %comm : tensor<4xf32>, !comm.mpi.request
    return %0#0, %0#1 : tensor<4xf32>, !comm.mpi.request
}
