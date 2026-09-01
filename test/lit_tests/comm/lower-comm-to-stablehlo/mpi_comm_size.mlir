// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: tensor<i64>) -> tensor<i32> {
func.func @main(%comm : !comm.mpi_comm) -> tensor<i32> {
    // CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiCommSize(%[[COMM]]) : (tensor<i64>) -> tensor<i32>
    %0 = comm.mpi.comm_size %comm : tensor<i32>
    return %0 : tensor<i32>
}
