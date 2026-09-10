// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

func.func @main(%send_buffer: tensor<4xf64>, %dst: tensor<i32>,
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

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[SEND_BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]]: tensor<i32>, %[[SRC:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm) -> tensor<4xf64> {
// CHECK-NEXT: comm.nccl.group_start
// CHECK-NEXT: comm.nccl.send %[[SEND_BUFFER]], %[[DST]], %[[COMM]] : tensor<4xf64>
// CHECK-NEXT: %[[RECV_BUFFER:.*]] = comm.nccl.recv %[[SRC]], %[[COMM]] : tensor<4xf64>
// CHECK-NEXT: comm.nccl.group_end
// CHECK-NEXT: return %[[RECV_BUFFER]] : tensor<4xf64>
// CHECK-NEXT: }
