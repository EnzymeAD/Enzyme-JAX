// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[SRC:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> tensor<4xf64> {
func.func @main(%src: tensor<i32>, %tag: tensor<i32>, %comm: !comm.mpi.comm) -> tensor<4xf64> {
    // CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiRecv(%[[SRC]], %[[TAG]], %[[COMM]]) {has_side_effect = true} : (tensor<i32>, tensor<i32>, tensor<i64>) -> tensor<4xf64>
    %0 = comm.mpi.recv %src, %tag, %comm : (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> tensor<4xf64>
    return %0 : tensor<4xf64>
}
