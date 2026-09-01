// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[REQUEST:.*]]: !comm.mpi_request) {
func.func @main(%request : !comm.mpi_request) {
    // CHECK-NEXT: comm.mpi.wait %[[REQUEST]] : !comm.mpi_request
    comm.mpi.wait %request : !comm.mpi_request
    return
}
