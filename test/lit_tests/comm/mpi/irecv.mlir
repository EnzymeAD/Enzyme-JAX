// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK: func.func @main(%[[TAG:.*]]: i32, %[[SOURCE:.*]]: i32, %[[COMM:.*]]: !comm.mpi.comm) -> !comm.mpi.requested<tensor<4xf32>> {
func.func @main(%tag : i32, %source : i32, %comm : !comm.mpi.comm) -> !comm.mpi.requested<tensor<4xf32>> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.irecv %[[TAG]], %[[SOURCE]], %[[COMM]] : !comm.mpi.requested<tensor<4xf32>>
    %0 = comm.mpi.irecv %tag, %source, %comm : !comm.mpi.requested<tensor<4xf32>>
    return %0 : !comm.mpi.requested<tensor<4xf32>>
}
