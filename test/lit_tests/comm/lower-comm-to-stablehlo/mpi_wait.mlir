// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[REQ:.*]]: tensor<i64>) {
func.func @main(%req : !comm.mpi_request) {
    // CHECK-NEXT: stablehlo.custom_call @MpiWait(%[[REQ]]) {has_side_effect = true} : (tensor<i64>) -> ()
    comm.mpi.wait %req : !comm.mpi_request
    return
}
