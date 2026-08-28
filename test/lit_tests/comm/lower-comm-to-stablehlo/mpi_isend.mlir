// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]] : tensor<i32>, %[[TAG:.*]] : tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> tensor<i64> {
func.func @main(%buffer: tensor<4xf64>, %dst: tensor<i32>, %tag: tensor<i32>, %comm: !comm.mpi.comm) -> !comm.mpi.request {
    // CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiIsend(%[[BUFFER]], %[[DST]], %[[TAG]], %[[COMM]]) {has_side_effect = true} : (tensor<4xf64>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<i64>)
    %0 = comm.mpi.isend %buffer, %dst, %tag, %comm : (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
    return %0 : !comm.mpi.request
}
