// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[REQUESTED:.*]]: !comm.mpi.requested<tensor<4xf32>>) -> tensor<4xf32> {
func.func @main(%requested : !comm.mpi.requested<tensor<4xf32>>) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.wait %[[REQUESTED]] : (!comm.mpi.requested<tensor<4xf32>>) -> tensor<4xf32>
    %0 = comm.mpi.wait %requested : (!comm.mpi.requested<tensor<4xf32>>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
