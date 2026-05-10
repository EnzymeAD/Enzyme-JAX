// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%buf : tensor<4xf32>, %tag : i32, %dest : i32, %comm : !comm.mpi.comm) -> !comm.mpi.request {
    // CHECK-NEXT: comm.mpi.send %buf, %tag, %dest, %comm : tensor<4xf32>, i32, i32, !comm.mpi.comm
    %0 = comm.mpi.isend %buf, %tag, %dest, %comm : (tensor<4xf32>, i32, i32, !comm.mpi.comm) -> !comm.mpi.request
    return %0 : !comm.mpi.request
}
