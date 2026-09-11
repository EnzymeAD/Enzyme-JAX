// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[PEER:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm) {
func.func @main(%buf : tensor<4xf32>, %peer : tensor<i32>, %comm : !comm.nccl.comm) {
    // CHECK-NEXT: comm.nccl.send %[[BUF]], %[[PEER]], %[[COMM]] : tensor<4xf32>
    comm.nccl.send %buf, %peer, %comm : tensor<4xf32>
    return
}
