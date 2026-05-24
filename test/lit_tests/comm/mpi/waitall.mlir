// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[REQ1:.*]]: !comm.mpi.requested<tensor<4xf32>>, %[[REQ2:.*]]: !comm.mpi.requested<tensor<5x3xf32>>) -> (tensor<4xf32>, tensor<5x3xf32>) {
func.func @main(%req1 : !comm.mpi.requested<tensor<4xf32>>, %req2 : !comm.mpi.requested<tensor<5x3xf32>>) -> (tensor<4xf32>, tensor<5x3xf32>) {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.waitall %[[REQ1]], %[[REQ2]] : (!comm.mpi.requested<tensor<4xf32>>, !comm.mpi.requested<tensor<5x3xf32>>) -> (tensor<4xf32>, tensor<5x3xf32>)
    %0:2 = comm.mpi.waitall %req1, %req2 : (!comm.mpi.requested<tensor<4xf32>>, !comm.mpi.requested<tensor<5x3xf32>>) -> (tensor<4xf32>, tensor<5x3xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<5x3xf32>
}
