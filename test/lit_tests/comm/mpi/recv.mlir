// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[TAG:.*]]: i32, %[[SOURCE:.*]]: i32, %[[COMM:.*]]: !comm.mpi.comm) -> tensor<4xf32> {
func.func @main(%tag : i32, %source : i32, %comm : !comm.mpi.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.recv %[[TAG]], %[[SOURCE]], %[[COMM]] : tensor<4xf32>
    %0 = comm.mpi.recv %tag, %source, %comm : tensor<4xf32>
    return %0 : tensor<4xf32>
}
