// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%buf : tensor<4xf32>, %comm : !comm.mpi.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.allreduce %buf, #comm.mpi.op<MPI_OP_NULL>, %comm : (tensor<4xf32>) -> tensor<4xf32>
    %0 = comm.mpi.allreduce %buf, #comm.mpi.op<MPI_OP_NULL>, %comm : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
