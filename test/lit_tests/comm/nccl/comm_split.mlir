// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.nccl.comm, %[[COLOR:.*]]: tensor<i32>, %[[KEY:.*]]: tensor<i32>) -> !comm.nccl.comm {
func.func @main(%comm : !comm.nccl.comm, %color : tensor<i32>, %key : tensor<i32>) -> !comm.nccl.comm {
    // CHECK-NEXT: [[v0:%.*]] = comm.nccl.comm_split %[[COMM]], %[[COLOR]], %[[KEY]] : (!comm.nccl.comm) -> !comm.nccl.comm
    %0 = comm.nccl.comm_split %comm, %color, %key : (!comm.nccl.comm) -> !comm.nccl.comm
    return %0 : !comm.nccl.comm
}
