// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi.comm, %[[COLOR:.*]]: i32, %[[KEY:.*]]: i32) -> !comm.mpi.comm {
func.func @main(%comm : !comm.mpi.comm, %color : i32, %key : i32) -> !comm.mpi.comm {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.comm_split %[[COMM]], %[[COLOR]], %[[KEY]] : (!comm.mpi.comm) -> !comm.mpi.comm
    %0 = comm.mpi.comm_split %comm, %color, %key : (!comm.mpi.comm) -> !comm.mpi.comm
    return %0 : !comm.mpi.comm
}
