// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: tensor<i64>, %[[REQ:.*]]: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
func.func @main(%comm : !comm.mpi.comm, %req : !comm.mpi.request) -> (!comm.mpi.comm, !comm.mpi.request) {
    // CHECK-NEXT: return %[[v0:.*]], %[[v1:.*]] : tensor<i64>, tensor<i64>
    return %comm, %req : !comm.mpi.comm, !comm.mpi.request
}
