// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

// NCCL pairs sends and recvs between two ranks by issue order and ignores
// tags, so we stable-sort sends and recvs inside the group by tag.
// Both ranks then agree on the order, whatever order they were posted in.

// A two-sided exchange with the same peer (2-rank halo exchange pattern)
func.func @same_peer_exchange(%send_west: tensor<4xf64>, %send_east: tensor<8xf64>,
                              %peer: tensor<i32>, %comm: !comm.mpi.comm)
    -> (tensor<8xf64>, tensor<4xf64>) {
  %c10 = stablehlo.constant dense<10> : tensor<i32>
  %c11 = stablehlo.constant dense<11> : tensor<i32>
  %recv_west, %recv_west_req = comm.mpi.irecv %peer, %c11, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> (tensor<8xf64>, !comm.mpi.request)
  %recv_east, %recv_east_req = comm.mpi.irecv %peer, %c10, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> (tensor<4xf64>, !comm.mpi.request)
  %send_west_req = comm.mpi.isend %send_west, %peer, %c10, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %send_east_req = comm.mpi.isend %send_east, %peer, %c11, %comm :
      (tensor<8xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.waitall %recv_west_req, %recv_east_req, %send_west_req, %send_east_req :
      !comm.mpi.request, !comm.mpi.request, !comm.mpi.request, !comm.mpi.request
  return %recv_west, %recv_east : tensor<8xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @same_peer_exchange
// CHECK-SAME: (%[[SEND_WEST:.*]]: tensor<4xf64>, %[[SEND_EAST:.*]]: tensor<8xf64>, %[[PEER:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm)
// CHECK: %[[GROUP:.*]]:2 = comm.nccl.group -> tensor<4xf64>, tensor<8xf64> {
// CHECK-NEXT: comm.nccl.send %[[SEND_WEST]], %[[PEER]], %[[COMM]] : tensor<4xf64>
// CHECK-NEXT: comm.nccl.send %[[SEND_EAST]], %[[PEER]], %[[COMM]] : tensor<8xf64>
// CHECK-NEXT: %[[RECV_EAST:.*]] = comm.nccl.recv %[[PEER]], %[[COMM]] {{.*}} -> tensor<4xf64>
// CHECK-NEXT: %[[RECV_WEST:.*]] = comm.nccl.recv %[[PEER]], %[[COMM]] {{.*}} -> tensor<8xf64>
// CHECK-NEXT: comm.nccl.yield %[[RECV_EAST]], %[[RECV_WEST]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GROUP]]#1, %[[GROUP]]#0 : tensor<8xf64>, tensor<4xf64>

// Equal tags keep their posting order (MPI's non-overtaking rule); only the
// tag-7 send moves ahead of them.
func.func @stable_equal_tags(%a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>,
                             %peer: tensor<i32>, %comm: !comm.mpi.comm) {
  %c7 = stablehlo.constant dense<7> : tensor<i32>
  %c9 = stablehlo.constant dense<9> : tensor<i32>
  %ra = comm.mpi.isend %a, %peer, %c9, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %rb = comm.mpi.isend %b, %peer, %c9, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %rc = comm.mpi.isend %c, %peer, %c7, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.waitall %ra, %rb, %rc : !comm.mpi.request, !comm.mpi.request, !comm.mpi.request
  return
}

// CHECK-LABEL: func.func @stable_equal_tags
// CHECK-SAME: (%[[A:.*]]: tensor<4xf64>, %[[B:.*]]: tensor<4xf64>, %[[C:.*]]: tensor<4xf64>,
// CHECK: comm.nccl.group {
// CHECK-NEXT: comm.nccl.send %[[C]],
// CHECK-NEXT: comm.nccl.send %[[A]],
// CHECK-NEXT: comm.nccl.send %[[B]],
// CHECK-NEXT: }

// Sends to different peers are sorted together by tag; each peer still sees
// its own sends in tag order.
func.func @multiple_peers(%a: tensor<4xf64>, %b: tensor<4xf64>,
                          %west: tensor<i32>, %east: tensor<i32>, %comm: !comm.mpi.comm) {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %ra = comm.mpi.isend %a, %west, %c2, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  %rb = comm.mpi.isend %b, %east, %c1, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.waitall %ra, %rb : !comm.mpi.request, !comm.mpi.request
  return
}

// CHECK-LABEL: func.func @multiple_peers
// CHECK-SAME: (%[[A:.*]]: tensor<4xf64>, %[[B:.*]]: tensor<4xf64>, %[[WEST:.*]]: tensor<i32>, %[[EAST:.*]]: tensor<i32>,
// CHECK: comm.nccl.group {
// CHECK-NEXT: comm.nccl.send %[[B]], %[[EAST]],
// CHECK-NEXT: comm.nccl.send %[[A]], %[[WEST]],
