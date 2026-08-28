// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi.comm, %[[COLOR:.*]]: tensor<i32>, %[[KEY:.*]]: tensor<i32>) -> !comm.mpi.comm {
func.func @main(%comm : !comm.mpi.comm, %color : tensor<i32>, %key : tensor<i32>) -> !comm.mpi.comm {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.comm_split %[[COMM]], %[[COLOR]], %[[KEY]] : (!comm.mpi.comm, tensor<i32>, tensor<i32>) -> !comm.mpi.comm
    %0 = comm.mpi.comm_split %comm, %color, %key : (!comm.mpi.comm, tensor<i32>, tensor<i32>) -> !comm.mpi.comm
    return %0 : !comm.mpi.comm
}
