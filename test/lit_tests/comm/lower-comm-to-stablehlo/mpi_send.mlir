// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) {
func.func @main(%buffer: tensor<4xf64>, %dst: tensor<i32>, %tag: tensor<i32>, %comm: !comm.mpi_comm) {
    // CHECK-NEXT: stablehlo.custom_call @MpiSend(%[[BUFFER]], %[[DST]], %[[TAG]], %[[COMM]]) {has_side_effect = true} : (tensor<4xf64>, tensor<i32>, tensor<i32>, tensor<i64>) -> ()
    comm.mpi.send %buffer, %dst, %tag, %comm : tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi_comm
    return
}
