// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[REQ1:.*]]: !comm.mpi_request, %[[REQ2:.*]]: !comm.mpi_request) {
func.func @main(%req1 : !comm.mpi_request, %req2 : !comm.mpi_request) {
    // CHECK-NEXT: comm.mpi.waitall %[[REQ1]], %[[REQ2]] : !comm.mpi_request, !comm.mpi_request
    comm.mpi.waitall %req1, %req2 : !comm.mpi_request, !comm.mpi_request
    return
}
