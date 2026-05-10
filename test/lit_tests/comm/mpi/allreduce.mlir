// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%buf : tensor<4xf32>, %op : !comm.mpi.op, %comm : !comm.mpi.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[v0:.*]] = comm.mpi.allreduce %buf, %op, %comm : (tensor<4xf32>, !comm.mpi.op, !comm.mpi.comm) -> tensor<4xf32>
    %0 = comm.mpi.allreduce %buf, %op, %comm : (tensor<4xf32>, !comm.mpi.op, !comm.mpi.comm) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
