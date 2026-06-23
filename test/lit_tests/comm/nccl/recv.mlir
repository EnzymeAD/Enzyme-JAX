// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[peer:.*]]: i32, %[[COMM:.*]]: !comm.nccl.comm, %[[STREAM:.*]]: !comm.nccl.stream) -> tensor<4xf32> {
func.func @main(%peer : i32, %comm : !comm.nccl.comm, %stream : !comm.nccl.stream) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.nccl.recv %[[peer]], %[[COMM]], %[[STREAM]] : tensor<4xf32>
    %0 = comm.nccl.recv %peer, %comm, %stream : tensor<4xf32>
    return %0 : tensor<4xf32>
}
