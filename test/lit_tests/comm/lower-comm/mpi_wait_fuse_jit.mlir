// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl --lower-comm-to-jit --fuse-jit="dag=false adjacent=true" %s | FileCheck %s --check-prefix=NCCL
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_Isend=0x1,MPI_Irecv=0x2,MPI_Wait=0x3,MPI_DOUBLE=0x4,MPI_STATUS_IGNORE=0x0" --lower-comm-to-jit --fuse-jit="dag=false adjacent=true" %s | FileCheck %s --check-prefix=MPI

// Each mpi.wait completes its request separately. Without fusion, XLA is free
// to reorder the resulting calls, which can deadlock against the peer.
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

// NCCL-LABEL: llvm.func @fused__enzymexla_jitwrap_ncclSend_enzymexla_jitwrap_ncclRecv
// NCCL-SAME: attributes {enzymexla.comm}
// NCCL: llvm.call @ncclSend
// NCCL: llvm.call @ncclRecv

// NCCL-LABEL: func.func @main
// NCCL-NOT: enzymexla.jit_call
// NCCL: %[[RESULT:.*]] = enzymexla.jit_call @fused__enzymexla_jitwrap_ncclSend_enzymexla_jitwrap_ncclRecv
// NCCL-NOT: enzymexla.jit_call
// NCCL: return %[[RESULT]] : tensor<4xf64>

// MPI-LABEL: func.func @main
// MPI-NOT: enzymexla.jit_call
// MPI: %[[RESULT:.*]]:{{[0-9]+}} = enzymexla.jit_call @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Wait_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Wait
// MPI-NOT: enzymexla.jit_call
// MPI: return %[[RESULT]]#0 : tensor<4xf64>
