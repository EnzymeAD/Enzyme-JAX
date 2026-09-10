// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

func.func @isend(%buffer: tensor<4xf64>, %dst: tensor<i32>,
                 %tag: tensor<i32>, %comm: !comm.mpi.comm) {
  %request = comm.mpi.isend %buffer, %dst, %tag, %comm :
      (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
  comm.mpi.wait %request : !comm.mpi.request
  return
}

func.func @irecv(%src: tensor<i32>, %tag: tensor<i32>,
                 %comm: !comm.mpi.comm) -> tensor<4xf64> {
  %buffer, %request = comm.mpi.irecv %src, %tag, %comm :
      (tensor<i32>, tensor<i32>, !comm.mpi.comm) ->
      (tensor<4xf64>, !comm.mpi.request)
  comm.mpi.wait %request : !comm.mpi.request
  return %buffer : tensor<4xf64>
}

// CHECK-LABEL: func.func @isend
// CHECK-SAME: (%[[BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm) {
// CHECK-NEXT: comm.nccl.send %[[BUFFER]], %[[DST]], %[[COMM]] : tensor<4xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @irecv
// CHECK-SAME: (%[[SRC:.*]]: tensor<i32>, %[[TAG2:.*]]: tensor<i32>, %[[COMM2:.*]]: !comm.nccl.comm) -> tensor<4xf64> {
// CHECK-NEXT: %[[BUFFER2:.*]] = comm.nccl.recv %[[SRC]], %[[COMM2]] : tensor<4xf64>
// CHECK-NEXT: return %[[BUFFER2]] : tensor<4xf64>
// CHECK-NEXT: }
