// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%buf : tensor<4xf32>, %root : i32, %comm : !comm.mpi.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.bcast %buf, %root, %comm : (tensor<4xf32>, i32, !comm.mpi.comm) -> tensor<4xf32>
    %0 = comm.mpi.bcast %buf, %root, %comm : (tensor<4xf32>, i32, !comm.mpi.comm) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
