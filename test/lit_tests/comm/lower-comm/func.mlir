// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%comm : !comm.mpi.comm, %req : !comm.mpi.request) -> (!comm.mpi.comm, !comm.mpi.request) {
    return %comm, %req : !comm.mpi.comm, !comm.mpi.request
}

// SHLO: func.func @main(%[[COMM:.*]]: tensor<i64>, %[[REQ:.*]]: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
// SHLO-NEXT: return %[[v0:.*]], %[[v1:.*]] : tensor<i64>, tensor<i64>

// JIT: func.func @main(%[[COMM:.*]]: tensor<i64>, %[[REQ:.*]]: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
// JIT-NEXT: return %[[v0:.*]], %[[v1:.*]] : tensor<i64>, tensor<i64>
