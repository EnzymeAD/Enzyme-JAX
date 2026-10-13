// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.nccl.comm) -> tensor<i32> {
func.func @main(%comm : !comm.nccl.comm) -> tensor<i32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.nccl.comm_user_rank %[[COMM]] : tensor<i32>
    %0 = comm.nccl.comm_user_rank %comm : tensor<i32>
    return %0 : tensor<i32>
}
