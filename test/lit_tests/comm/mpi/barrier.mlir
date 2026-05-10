// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%comm : !comm.mpi.comm) {
    // CHECK-NEXT: comm.mpi.barrier %comm
    comm.mpi.barrier %comm
    return
}
