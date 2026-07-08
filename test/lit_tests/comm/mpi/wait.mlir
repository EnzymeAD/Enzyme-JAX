// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[REQUEST:.*]]: !comm.mpi.request) {
func.func @main(%request : !comm.mpi.request) {
    // CHECK-NEXT: comm.mpi.wait %[[REQUEST]] : !comm.mpi.request
    comm.mpi.wait %request : !comm.mpi.request
    return
}
