// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%comm : !comm.mpi.comm, %color : i32, %key : i32) -> !comm.mpi.comm {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.comm_split %comm, %color, %key : !comm.mpi.comm
    %0 = comm.mpi.comm_split %comm, %color, %key : (!comm.mpi.comm, i32, i32) -> !comm.mpi.comm
    return %0 : !comm.mpi.comm
}
