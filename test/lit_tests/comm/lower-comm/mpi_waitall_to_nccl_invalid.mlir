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

// -----

// Two sends with a non-constant tag can't be put in a canonical order, and
// NCCL would otherwise pair them by posting order and ignore the tags.
func.func @dynamic_tags(%a: tensor<4xf64>, %b: tensor<4xf64>, %peer: tensor<i32>,
                        %tag: tensor<i32>, %comm: !comm.mpi.comm) {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %ra = comm.mpi.isend %a, %peer, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %rb = comm.mpi.isend %b, %peer, %c1, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  // expected-error @+1 {{mpi.waitall is not in a form supported by MPI-to-NCCL legalization}}
  comm.mpi.waitall %ra, %rb : !comm.mpi.request, !comm.mpi.request
  return
}
