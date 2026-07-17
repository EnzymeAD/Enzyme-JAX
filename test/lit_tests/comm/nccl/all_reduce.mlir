// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf32>, %[[COMM:.*]]: !comm.nccl.comm, %[[STREAM:.*]]: !comm.nccl.stream)  -> tensor<4xf32> {
func.func @main(%buf : tensor<4xf32>, %comm : !comm.nccl.comm, %stream : !comm.nccl.stream) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.nccl.all_reduce %[[BUF]], <NCCL_SUM>, %[[COMM]], %[[STREAM]] : (tensor<4xf32>) -> tensor<4xf32>
    %0 = comm.nccl.all_reduce %buf, #comm.nccl.red_op<NCCL_SUM>, %comm, %stream : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
