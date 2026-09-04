// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>) {
func.func @main(%req1 : !comm.mpi.request, %req2 : !comm.mpi.request) {
    // CHECK-NEXT: stablehlo.custom_call @MpiWaitall(%[[REQ1]], %[[REQ2]]) {has_side_effect = true} : (tensor<i64>, tensor<i64>) -> ()
    comm.mpi.waitall %req1, %req2 : !comm.mpi.request, !comm.mpi.request
    return
}
