// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[TAG:.*]]: i32, %[[DEST:.*]]: i32, %[[COMM:.*]]: !comm.mpi.comm) -> !comm.mpi.request {
func.func @main(%buf : tensor<4xf32>, %tag : i32, %dest : i32, %comm : !comm.mpi.comm) -> !comm.mpi.request {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.isend %[[BUF]], %[[TAG]], %[[DEST]], %[[COMM]] : (tensor<4xf32>) -> !comm.mpi.request
    %0 = comm.mpi.isend %buf, %tag, %dest, %comm : (tensor<4xf32>) -> !comm.mpi.request
    return %0 : !comm.mpi.request
}
