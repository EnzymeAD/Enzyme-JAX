// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

// The group is placed at the mpi.waitall, after unrelated computation that
// sits between the request producers.
func.func @main(%send_buffer: tensor<4xf64>, %peer: tensor<i32>,
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

// CHECK-LABEL: func.func @main
// CHECK-NEXT: %[[DOUBLED:.*]] = stablehlo.add
// CHECK-NEXT: %[[RECV_BUFFER:.*]] = comm.nccl.group -> tensor<4xf64> {
// CHECK-NEXT: comm.nccl.send
// CHECK-NEXT: %[[RECV:.*]] = comm.nccl.recv
// CHECK-NEXT: comm.nccl.yield %[[RECV]] : tensor<4xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[RECV_BUFFER]], %[[DOUBLED]]
// CHECK-NEXT: return %[[RECV_BUFFER]], %[[SUM]]
