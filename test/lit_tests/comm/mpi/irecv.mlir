// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%tag : i32, %source : i32, %comm : !comm.mpi.comm) -> !comm.mpi.requested<tensor<4xf32>> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.irecv %tag, %source, %comm : !comm.mpi.requested<tensor<4xf32>>
    %0 = comm.mpi.irecv %tag, %source, %comm : !comm.mpi.requested<tensor<4xf32>>
    return %0 : !comm.mpi.requested<tensor<4xf32>>
}
