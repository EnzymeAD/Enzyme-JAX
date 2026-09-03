// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: tensor<i64>) -> tensor<i32> {
func.func @main(%comm : !comm.mpi.comm) -> tensor<i32> {
// CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiCommRank(%[[COMM]]) : (tensor<i64>) -> tensor<i32>
    %0 = comm.mpi.comm_rank %comm : tensor<i32>
    return %0 : tensor<i32>
}
