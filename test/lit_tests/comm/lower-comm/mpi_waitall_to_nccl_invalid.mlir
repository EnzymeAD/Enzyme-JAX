// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl -split-input-file -verify-diagnostics %s

// A blocking comm op between a producer and its mpi.waitall would change the
// relative order of communication.
func.func @interleaved_comm(%buffer: tensor<4xf64>, %dst: tensor<i32>,
                            %tag: tensor<i32>, %comm: !comm.mpi.comm) {
  %request = comm.mpi.isend %buffer, %dst, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.send %buffer, %dst, %tag, %comm :
      tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm
  // expected-error @+1 {{mpi.waitall is not in a form supported by MPI-to-NCCL legalization}}
  comm.mpi.waitall %request : !comm.mpi.request
  return
}

// -----

// A received buffer used before its mpi.waitall cannot be moved into a group.
func.func @early_use(%peer: tensor<i32>, %tag: tensor<i32>,
                     %comm: !comm.mpi.comm) -> tensor<4xf64> {
  %recv_buffer, %recv_request = comm.mpi.irecv %peer, %tag, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) ->
      (tensor<4xf64>, !comm.mpi.request)
  %early = stablehlo.add %recv_buffer, %recv_buffer : tensor<4xf64>
  // expected-error @+1 {{mpi.waitall is not in a form supported by MPI-to-NCCL legalization}}
  comm.mpi.waitall %recv_request : !comm.mpi.request
  return %early : tensor<4xf64>
}
