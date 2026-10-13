// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[peer:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm) -> tensor<4xf32> {
func.func @main(%peer : tensor<i32>, %comm : !comm.nccl.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.nccl.recv %[[peer]], %[[COMM]] : tensor<4xf32>
    %0 = comm.nccl.recv %peer, %comm : tensor<4xf32>
    return %0 : tensor<4xf32>
}
