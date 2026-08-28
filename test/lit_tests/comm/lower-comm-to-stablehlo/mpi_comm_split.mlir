// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: tensor<i64>, %[[COLOR:.*]] : tensor<i32>, %[[KEY:.*]] : tensor<i32>) -> tensor<i64> {
func.func @main(%comm : !comm.mpi.comm, %color : tensor<i32>, %key : tensor<i32>) -> !comm.mpi.comm {
    // CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiCommSplit(%[[COMM]], %[[COLOR]], %[[KEY]]) : (tensor<i64>, tensor<i32>, tensor<i32>) -> tensor<i64>
    %0 = comm.mpi.comm_split %comm, %color, %key : (!comm.mpi.comm, tensor<i32>, tensor<i32>) -> !comm.mpi.comm
    return %0 : !comm.mpi.comm
}
