// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%comm : !comm.mpi.comm) -> i32 {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.comm_rank %comm : i32
    %0 = comm.mpi.comm_rank %comm : i32
    return %0 : i32
}
