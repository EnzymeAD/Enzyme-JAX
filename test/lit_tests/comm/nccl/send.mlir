// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[PEER:.*]]: i32, %[[COMM:.*]]: !comm.nccl.comm, %[[STREAM:.*]]: !comm.nccl.stream) {
func.func @main(%buf : tensor<4xf32>, %peer : i32, %comm : !comm.nccl.comm, %stream : !comm.nccl.stream) {
    // CHECK-NEXT: comm.nccl.send %[[BUF]], %[[PEER]], %[[COMM]], %[[STREAM]] : tensor<4xf32>
    comm.nccl.send %buf, %peer, %comm, %stream : tensor<4xf32>
    return
}
