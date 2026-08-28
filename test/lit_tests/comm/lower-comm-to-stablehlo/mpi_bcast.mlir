// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[IN_BUFFER:.*]]: tensor<4xf64>, %[[ROOT:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> tensor<4xf64> {
func.func @main(%inBuffer : tensor<4xf64>, %root : tensor<i32>, %comm : !comm.mpi.comm) -> tensor<4xf64> {
    // CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiBcast(%[[IN_BUFFER]], %[[ROOT]], %[[COMM]]) {has_side_effect = true} : (tensor<4xf64>, tensor<i32>, tensor<i64>) -> tensor<4xf64>
    %0 = comm.mpi.bcast %inBuffer, %root, %comm : (tensor<4xf64>, tensor<i32>, !comm.mpi.comm) -> (tensor<4xf64>)
    return %0 : tensor<4xf64>
}
