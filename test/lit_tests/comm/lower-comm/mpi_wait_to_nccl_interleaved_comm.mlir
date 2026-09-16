// RUN: not enzymexlamlir-opt --legalize-mpi-to-nccl %s 2>&1 | FileCheck %s

func.func @main(%buffer: tensor<4xf64>, %dst: tensor<i32>,
                %tag: tensor<i32>, %comm: !comm.mpi.comm) {
  %request = comm.mpi.isend %buffer, %dst, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.send %buffer, %dst, %tag, %comm :
      tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm
  comm.mpi.wait %request : !comm.mpi.request
  return
}

// CHECK: error: mpi.wait is not in a form supported by MPI-to-NCCL legalization
