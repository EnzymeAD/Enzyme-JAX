// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @send_recv
// CHECK-SAME: (%[[BUF:.*]]: tensor<4xf32>, %[[PEER:.*]]: tensor<i32>, %[[COMM:.*]]: !comm.nccl.comm) -> tensor<4xf32> {
func.func @send_recv(%buf : tensor<4xf32>, %peer : tensor<i32>, %comm : !comm.nccl.comm) -> tensor<4xf32> {
    // CHECK-NEXT: %[[RESULT:.*]] = comm.nccl.group -> tensor<4xf32> {
    // CHECK-NEXT: comm.nccl.send %[[BUF]], %[[PEER]], %[[COMM]] : tensor<4xf32>, tensor<i32>, !comm.nccl.comm
    // CHECK-NEXT: %[[RECV:.*]] = comm.nccl.recv %[[PEER]], %[[COMM]] : (tensor<i32>, !comm.nccl.comm) -> tensor<4xf32>
    // CHECK-NEXT: comm.nccl.yield %[[RECV]] : tensor<4xf32>
    // CHECK-NEXT: }
    %result = comm.nccl.group -> tensor<4xf32> {
      comm.nccl.send %buf, %peer, %comm : tensor<4xf32>, tensor<i32>, !comm.nccl.comm
      %recv = comm.nccl.recv %peer, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<4xf32>
      comm.nccl.yield %recv : tensor<4xf32>
    }
    // CHECK-NEXT: return %[[RESULT]] : tensor<4xf32>
    return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @send_only
func.func @send_only(%buf : tensor<4xf32>, %peer : tensor<i32>, %comm : !comm.nccl.comm) {
    // CHECK-NEXT: comm.nccl.group {
    // CHECK-NEXT: comm.nccl.send
    // CHECK-NEXT: }
    comm.nccl.group {
      comm.nccl.send %buf, %peer, %comm : tensor<4xf32>, tensor<i32>, !comm.nccl.comm
    }
    return
}

// CHECK-LABEL: func.func @two_recvs
func.func @two_recvs(%peer0 : tensor<i32>, %peer1 : tensor<i32>, %comm : !comm.nccl.comm) -> (tensor<4xf32>, tensor<2xi32>) {
    // CHECK-NEXT: %{{.*}}:2 = comm.nccl.group -> tensor<4xf32>, tensor<2xi32> {
    // CHECK: comm.nccl.yield %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<2xi32>
    %results:2 = comm.nccl.group -> tensor<4xf32>, tensor<2xi32> {
      %0 = comm.nccl.recv %peer0, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<4xf32>
      %1 = comm.nccl.recv %peer1, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<2xi32>
      comm.nccl.yield %0, %1 : tensor<4xf32>, tensor<2xi32>
    }
    return %results#0, %results#1 : tensor<4xf32>, tensor<2xi32>
}
