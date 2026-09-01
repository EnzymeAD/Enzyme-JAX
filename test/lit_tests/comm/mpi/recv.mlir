// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: func.func @main(%[[TAG:.*]]: tensor<i32>, %[[SOURCE:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.mpi.comm) -> tensor<4xf32> {
func.func @main(%tag : tensor<i32>, %source : tensor<i32>, %comm : !comm.mpi.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.recv %[[TAG]], %[[SOURCE]], %[[COMM]] : (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> tensor<4xf32>
    %0 = comm.mpi.recv %tag, %source, %comm : (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
