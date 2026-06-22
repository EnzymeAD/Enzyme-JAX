// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[REQ1:.*]]: !comm.mpi.request, %[[REQ2:.*]]: !comm.mpi.request) {
func.func @main(%req1 : !comm.mpi.request, %req2 : !comm.mpi.request) {
    // CHECK-NEXT: comm.mpi.waitall %[[REQ1]], %[[REQ2]] : !comm.mpi.request, !comm.mpi.request
    comm.mpi.waitall %req1, %req2 : !comm.mpi.request, !comm.mpi.request
    return
}
