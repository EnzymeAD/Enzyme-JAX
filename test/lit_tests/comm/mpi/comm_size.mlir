// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi.comm) -> i32 {
func.func @main(%comm : !comm.mpi.comm) -> i32 {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.comm_size %[[COMM]] : i32
    %0 = comm.mpi.comm_size %comm : i32
    return %0 : i32
}
