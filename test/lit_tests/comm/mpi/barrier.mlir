// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi.comm) {
func.func @main(%comm : !comm.mpi.comm) {
    // CHECK-NEXT: comm.mpi.barrier %[[COMM]]
    comm.mpi.barrier %comm
    return
}
