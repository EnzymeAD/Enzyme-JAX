// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%tag : i32, %source : i32, %comm : !comm.mpi.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.recv %tag, %source, %comm : (i32, i32, !comm.mpi.comm) -> tensor<4xf32>
    %0 = comm.mpi.recv %tag, %source, %comm : (i32, i32, !comm.mpi.comm) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
