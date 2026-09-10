// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

func.func @main(%buffer: tensor<4xf64>,
                %comm: !comm.mpi.comm) -> tensor<4xf64> {
  %result = comm.mpi.allreduce %buffer, #comm.mpi.op<MPI_SUM>, %comm :
      (tensor<4xf64>, !comm.mpi.comm) -> tensor<4xf64>
  return %result : tensor<4xf64>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[BUFFER:.*]]: tensor<4xf64>, %[[COMM:.*]]: !comm.nccl.comm) -> tensor<4xf64> {
// CHECK-NEXT: %[[RESULT:.*]] = comm.nccl.all_reduce %[[BUFFER]], #comm.nccl.red_op<ncclSum>, %[[COMM]] : (tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[RESULT]] : tensor<4xf64>
// CHECK-NEXT: }
