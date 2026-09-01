// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: func.func @main(%[[COMM:.*]]: !comm.mpi_comm, %[[COLOR:.*]]: tensor<i32>, %[[KEY:.*]]: tensor<i32>) -> !comm.mpi_comm {
func.func @main(%comm : !comm.mpi_comm, %color : tensor<i32>, %key : tensor<i32>) -> !comm.mpi_comm {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.comm_split %[[COMM]], %[[COLOR]], %[[KEY]] : (!comm.mpi_comm, tensor<i32>, tensor<i32>) -> !comm.mpi_comm
    %0 = comm.mpi.comm_split %comm, %color, %key : (!comm.mpi_comm, tensor<i32>, tensor<i32>) -> !comm.mpi_comm
    return %0 : !comm.mpi_comm
}
