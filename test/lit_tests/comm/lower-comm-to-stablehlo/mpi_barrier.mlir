// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: tensor<i64>) {
func.func @main(%comm : !comm.mpi_comm) {
    // CHECK-NEXT: stablehlo.custom_call @MpiBarrier(%[[COMM]]) {has_side_effect = true} : (tensor<i64>) -> ()
    comm.mpi.barrier %comm : !comm.mpi_comm
    return
}
