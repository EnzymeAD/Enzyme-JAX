// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: func.func @main(%[[REQUEST:.*]]: !comm.mpi.request, %[[BUF:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @main(%request : !comm.mpi.request, %buf : tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.sync %[[REQUEST]], %[[BUF]] : tensor<4xf32>
    %0 = comm.mpi.sync %request, %buf : tensor<4xf32>
    return %0 : tensor<4xf32>
}
