// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[ROOT:.*]]: i32, %[[COMM:.*]]: !comm.nccl.comm, %[[STREAM:.*]]: !comm.nccl.stream)  -> tensor<4xf32> {
func.func @main(%buf : tensor<4xf32>, %root : i32, %comm : !comm.nccl.comm, %stream : !comm.nccl.stream) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.nccl.broadcast %[[BUF]], %[[ROOT]], %[[COMM]], %[[STREAM]] : (tensor<4xf32>) -> tensor<4xf32>
    %0 = comm.nccl.broadcast %buf, %root, %comm, %stream : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
