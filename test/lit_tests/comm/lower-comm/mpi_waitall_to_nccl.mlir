// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s
// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @send_recv(%send_buffer: tensor<4xf64>, %dst: tensor<i32>,
                     %src: tensor<i32>, %tag: tensor<i32>,
                     %comm: !comm.mpi.comm) -> tensor<4xf64> {
  %send_request = comm.mpi.isend %send_buffer, %dst, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %recv_buffer, %recv_request = comm.mpi.irecv %src, %tag, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) ->
      (tensor<4xf64>, !comm.mpi.request)
  comm.mpi.waitall %send_request, %recv_request : !comm.mpi.request, !comm.mpi.request
  return %recv_buffer : tensor<4xf64>
}

// The group is placed at the mpi.waitall, after unrelated computation that
// sits between the request producers.
func.func @moves_producers(%send_buffer: tensor<4xf64>, %peer: tensor<i32>,
                           %tag: tensor<i32>, %comm: !comm.mpi.comm) -> (tensor<4xf64>, tensor<4xf64>) {
  %send_request = comm.mpi.isend %send_buffer, %peer, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %doubled = stablehlo.add %send_buffer, %send_buffer : tensor<4xf64>
  %recv_buffer, %recv_request = comm.mpi.irecv %peer, %tag, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) ->
      (tensor<4xf64>, !comm.mpi.request)
  comm.mpi.waitall %send_request, %recv_request : !comm.mpi.request, !comm.mpi.request
  %sum = stablehlo.add %recv_buffer, %doubled : tensor<4xf64>
  return %recv_buffer, %sum : tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @send_recv
// CHECK-SAME: (%[[SEND_BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]]: tensor<i32>, %[[SRC:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm) -> tensor<4xf64> {
// CHECK-NEXT: %[[RECV_BUFFER:.*]] = comm.nccl.group -> tensor<4xf64> {
// CHECK-NEXT: comm.nccl.send %[[SEND_BUFFER]], %[[DST]], %[[COMM]] : tensor<4xf64>, tensor<i32>, !comm.nccl.comm
// CHECK-NEXT: %[[RECV:.*]] = comm.nccl.recv %[[SRC]], %[[COMM]] : (tensor<i32>, !comm.nccl.comm) -> tensor<4xf64>
// CHECK-NEXT: comm.nccl.yield %[[RECV]] : tensor<4xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RECV_BUFFER]] : tensor<4xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @moves_producers
// CHECK-NEXT: %[[DOUBLED:.*]] = stablehlo.add
// CHECK-NEXT: %[[RECV_BUFFER:.*]] = comm.nccl.group -> tensor<4xf64> {
// CHECK-NEXT: comm.nccl.send
// CHECK-NEXT: %[[RECV:.*]] = comm.nccl.recv
// CHECK-NEXT: comm.nccl.yield %[[RECV]] : tensor<4xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[RECV_BUFFER]], %[[DOUBLED]]
// CHECK-NEXT: return %[[RECV_BUFFER]], %[[SUM]]

// JIT-LABEL: func.func @send_recv
// JIT-NOT: enzymexla.jit_call
// JIT: %[[RESULT:.*]] = enzymexla.jit_call @enzymexla_jitwrap_ncclGroup_Send_Recv
// JIT-NOT: enzymexla.jit_call
// JIT: return %[[RESULT]] : tensor<4xf64>

// JIT-LABEL: func.func @moves_producers
// JIT-NOT: enzymexla.jit_call
// JIT: %[[RESULT:.*]] = enzymexla.jit_call @enzymexla_jitwrap_ncclGroup_Send_Recv
// JIT-NOT: enzymexla.jit_call
// JIT: return %[[RESULT]], %{{.*}}
