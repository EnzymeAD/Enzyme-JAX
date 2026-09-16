// RUN: not enzymexlamlir-opt --legalize-mpi-to-nccl %s 2>&1 | FileCheck %s

func.func @main(%buffer: tensor<4xf64>, %dst: tensor<i32>,
                %tag: tensor<i32>, %comm: !comm.mpi.comm) {
  %request = comm.mpi.isend %buffer, %dst, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.send %buffer, %dst, %tag, %comm :
      tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm
  comm.mpi.waitall %request : !comm.mpi.request
  return
}

// CHECK: error: expected single-use requests produced by comm.mpi.isend or comm.mpi.irecv in the same communication-safe block range
