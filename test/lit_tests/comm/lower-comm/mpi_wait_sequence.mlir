// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_Isend=0x1,MPI_Irecv=0x2,MPI_Wait=0x3,MPI_DOUBLE=0x4,MPI_STATUS_IGNORE=0x0" --lower-comm-to-jit --lower-jit="jit=false" %s | FileCheck %s --check-prefix=MPI
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_Isend=0x1,MPI_Irecv=0x2,MPI_Wait=0x3,MPI_DOUBLE=0x4,MPI_STATUS_IGNORE=0x0" --lower-comm-to-jit --fuse-jit="dag=false adjacent=true" --lower-jit="jit=false" %s | FileCheck %s --check-prefix=MPI-LOWERED
// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl --lower-comm-to-jit --lower-jit="jit=false" %s | FileCheck %s --check-prefix=NCCL
// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl --lower-comm-to-jit --fuse-jit="dag=false adjacent=true" --lower-jit="jit=false" %s | FileCheck %s --check-prefix=NCCL-LOWERED

// Each mpi.wait completes its request separately. If XLA reordered the
// resulting calls, they could deadlock against the peer. `lower-jit` prevents
// this by chaining every call marked with `enzymexla.require_jit_ordering`
// through a token, so the mark must survive fusion.
func.func @main(%send_buffer: tensor<4xf64>, %peer: tensor<i32>,
                %tag: tensor<i32>, %comm: !comm.mpi.comm) -> tensor<4xf64> {
  %send_request = comm.mpi.isend %send_buffer, %peer, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.wait %send_request : !comm.mpi.request
  %recv_buffer, %recv_request = comm.mpi.irecv %peer, %tag, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) ->
      (tensor<4xf64>, !comm.mpi.request)
  comm.mpi.wait %recv_request : !comm.mpi.request
  return %recv_buffer : tensor<4xf64>
}

// Without fusion, the token chains the four calls in program order.
// MPI-LABEL: func.func @main
// MPI: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// MPI-NOT: stablehlo.custom_call
// MPI: %[[SEND:.+]]:2 = stablehlo.custom_call @enzymexla_compile_cpu({{.*}}, %[[TOKEN]])
// MPI-SAME: -> (tensor<i64>, !stablehlo.token)
// MPI-NOT: stablehlo.custom_call
// MPI: %[[SEND_WAIT:.+]] = stablehlo.custom_call @enzymexla_compile_cpu(%[[SEND]]#0, %[[SEND]]#1)
// MPI-SAME: -> !stablehlo.token
// MPI-NOT: stablehlo.custom_call
// MPI: %[[RECV:.+]]:3 = stablehlo.custom_call @enzymexla_compile_cpu({{.*}}, %[[SEND_WAIT]])
// MPI-SAME: -> (tensor<4xf64>, tensor<i64>, !stablehlo.token)
// MPI-NOT: stablehlo.custom_call
// MPI: stablehlo.custom_call @enzymexla_compile_cpu(%[[RECV]]#1, %[[RECV]]#2)
// MPI-SAME: -> !stablehlo.token
// MPI-NOT: stablehlo.custom_call
// MPI: return %[[RECV]]#0 : tensor<4xf64>

// The fused call takes a token and returns it as its last result.
// MPI-LOWERED-LABEL: func.func @main
// MPI-LOWERED: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// MPI-LOWERED-NOT: stablehlo.custom_call
// MPI-LOWERED: %[[RESULT:.+]]:4 = stablehlo.custom_call @enzymexla_compile_cpu({{.*}}, %[[TOKEN]])
// MPI-LOWERED-SAME: output_tuple_indices = [3], operand_index = 11
// MPI-LOWERED-SAME: -> (tensor<4xf64>, tensor<i64>, tensor<i64>, !stablehlo.token)
// MPI-LOWERED-NOT: stablehlo.custom_call
// MPI-LOWERED: return %[[RESULT]]#0 : tensor<4xf64>

// Without fusion, the token chains the send and the receive in program order.
// NCCL-LABEL: func.func @main
// NCCL: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// NCCL-NOT: stablehlo.custom_call
// NCCL: %[[SEND:.+]] = stablehlo.custom_call @enzymexla_compile_gpu({{.*}}, %[[TOKEN]])
// NCCL-SAME: -> !stablehlo.token
// NCCL-NOT: stablehlo.custom_call
// NCCL: %[[RECV:.+]]:2 = stablehlo.custom_call @enzymexla_compile_gpu({{.*}}, %[[SEND]])
// NCCL-SAME: -> (tensor<4xf64>, !stablehlo.token)
// NCCL-NOT: stablehlo.custom_call
// NCCL: return %[[RECV]]#0 : tensor<4xf64>

// The fused call takes a token and returns it as its last result.
// NCCL-LOWERED-LABEL: func.func @main
// NCCL-LOWERED: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// NCCL-LOWERED-NOT: stablehlo.custom_call
// NCCL-LOWERED: %[[RESULT:.+]]:2 = stablehlo.custom_call @enzymexla_compile_gpu({{.*}}, %[[TOKEN]])
// NCCL-LOWERED-SAME: output_tuple_indices = [1], operand_index = 10
// NCCL-LOWERED-SAME: -> (tensor<4xf64>, !stablehlo.token)
// NCCL-LOWERED-NOT: stablehlo.custom_call
// NCCL-LOWERED: return %[[RESULT]]#0 : tensor<4xf64>

