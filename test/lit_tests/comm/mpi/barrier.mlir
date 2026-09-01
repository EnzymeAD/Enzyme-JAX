// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi_comm) {
func.func @main(%comm : !comm.mpi_comm) {
    // CHECK-NEXT: comm.mpi.barrier %[[COMM]] : !comm.mpi_comm
    comm.mpi.barrier %comm : !comm.mpi_comm
    return
}
