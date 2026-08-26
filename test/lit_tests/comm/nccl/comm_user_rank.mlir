// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.nccl.comm) -> i32 {
func.func @main(%comm : !comm.nccl.comm) -> i32 {
    // CHECK-NEXT: %[[v0:.*]] = comm.nccl.comm_user_rank %[[COMM]] : i32
    %0 = comm.nccl.comm_user_rank %comm : i32
    return %0 : i32
}
