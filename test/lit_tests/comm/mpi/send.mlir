// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[TAG:.*]]: tensor<i32>, %[[DEST:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.mpi.comm) {
func.func @main(%buf : tensor<4xf32>, %tag : tensor<i32>, %dest : tensor<i32>, %comm : !comm.mpi.comm) {
    // CHECK-NEXT: comm.mpi.send %[[BUF]], %[[TAG]], %[[DEST]], %[[COMM]] : tensor<4xf32>, tensor<i32>, tensor<i32>, !comm.mpi.comm
    comm.mpi.send %buf, %tag, %dest, %comm : tensor<4xf32>, tensor<i32>, tensor<i32>, !comm.mpi.comm
    return
}
