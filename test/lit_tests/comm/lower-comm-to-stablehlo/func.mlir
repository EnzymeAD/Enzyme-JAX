// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: tensor<i64>, %[[REQ:.*]]: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
func.func @main(%comm : !comm.mpi_comm, %req : !comm.mpi_request) -> (!comm.mpi_comm, !comm.mpi_request) {
    // CHECK-NEXT: return %[[v0:.*]], %[[v1:.*]] : tensor<i64>, tensor<i64>
    return %comm, %req : !comm.mpi_comm, !comm.mpi_request
}
