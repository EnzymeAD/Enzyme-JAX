// RUN: not enzymexlamlir-opt --legalize-mpi-to-nccl %s 2>&1 | FileCheck %s

// A received buffer used before its mpi.waitall cannot be moved into a group.
func.func @main(%peer: tensor<i32>, %tag: tensor<i32>,
                %comm: !comm.mpi.comm) -> tensor<4xf64> {
  %recv_buffer, %recv_request = comm.mpi.irecv %peer, %tag, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) ->
      (tensor<4xf64>, !comm.mpi.request)
  %early = stablehlo.add %recv_buffer, %recv_buffer : tensor<4xf64>
  comm.mpi.waitall %recv_request : !comm.mpi.request
  return %early : tensor<4xf64>
}

// CHECK: error: mpi.waitall is not in a form supported by MPI-to-NCCL legalization
