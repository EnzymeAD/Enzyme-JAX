// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[TAG:.*]]: i32, %[[DEST:.*]]: i32, %[[COMM:.*]]: !comm.mpi.comm) {
func.func @main(%buf : tensor<4xf32>, %tag : i32, %dest : i32, %comm : !comm.mpi.comm) {
    // CHECK-NEXT: comm.mpi.send %[[BUF]], %[[TAG]], %[[DEST]], %[[COMM]] : tensor<4xf32>
    comm.mpi.send %buf, %tag, %dest, %comm : tensor<4xf32>
    return
}
