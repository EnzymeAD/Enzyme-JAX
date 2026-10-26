// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi.comm) -> tensor<i32> {
func.func @main(%comm : !comm.mpi.comm) -> tensor<i32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.comm_size %[[COMM]] : tensor<i32>
    %0 = comm.mpi.comm_size %comm : tensor<i32>
    return %0 : tensor<i32>
}
