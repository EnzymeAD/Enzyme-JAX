// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: func.func @single(%[[REQ:.*]]: !comm.mpi.request, %[[BUF:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @single(%req : !comm.mpi.request, %buf : tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.syncall requests = (%[[REQ]]), buffers = (%[[BUF]]) : tensor<4xf32>
    %0 = comm.mpi.syncall requests = (%req), buffers = (%buf) : tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @multiple(%[[REQ1:.*]]: !comm.mpi.request, %[[REQ2:.*]]: !comm.mpi.request, %[[BUF1:.*]]: tensor<4xf32>, %[[BUF2:.*]]: tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>) {
func.func @multiple(%req1 : !comm.mpi.request, %req2 : !comm.mpi.request, %buf1 : tensor<4xf32>, %buf2 : tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>) {
    // CHECK-NEXT: %[[v0:.*]]:2 = comm.mpi.syncall requests = (%[[REQ1]], %[[REQ2]]), buffers = (%[[BUF1]], %[[BUF2]]) : tensor<4xf32>, tensor<8xi32>
    %0:2 = comm.mpi.syncall requests = (%req1, %req2), buffers = (%buf1, %buf2) : tensor<4xf32>, tensor<8xi32>
    return %0#0, %0#1 : tensor<4xf32>, tensor<8xi32>
}
