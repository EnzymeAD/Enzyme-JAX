// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[SRC:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> (tensor<4xf64>, tensor<i64>) {
func.func @main(%src: tensor<i32>, %tag: tensor<i32>, %comm: !comm.mpi_comm) -> (tensor<4xf64>, !comm.mpi_request) {
    // CHECK-NEXT: %[[v0:.*]]:2 = stablehlo.custom_call @MpiIrecv(%[[SRC]], %[[TAG]], %[[COMM]]) {has_side_effect = true} : (tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<4xf64>, tensor<i64>)
    %0:2 = comm.mpi.irecv %src, %tag, %comm : (tensor<i32>, tensor<i32>, !comm.mpi_comm) -> (tensor<4xf64>, !comm.mpi_request)
    return %0#0, %0#1 : tensor<4xf64>, !comm.mpi_request
}
