// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main(%buf: tensor<5xi32>, %peer: tensor<i32>) -> tensor<5xi32> {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  %result = comm.nccl.group -> tensor<5xi32> {
    comm.nccl.send %buf, %peer, %comm : tensor<5xi32>, tensor<i32>, !comm.nccl.comm
    %recv = comm.nccl.recv %peer, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<5xi32>
    comm.nccl.yield %recv : tensor<5xi32>
  }
  return %result : tensor<5xi32>
}

// The whole group is a single wrapper: all scalars are copied to the host and
// synchronized before ncclGroupStart, and the NCCL calls keep their order.
// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclGroup_Send_Recv
// CHECK-SAME: (%[[SEND_BUF:[^:]*]]: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %[[RECV_BUF:[^:]*]]: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %{{[^:]*}}: !llvm.ptr, %{{[^:]*}}: !llvm.ptr)
// CHECK-SAME: enzymexla.device_abi = "cuda"
// CHECK: %[[STREAM:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// CHECK: %[[SEND_PTR:.*]] = llvm.load %[[SEND_BUF]] : !llvm.ptr -> !llvm.ptr
// CHECK-COUNT-4: llvm.call @cuMemcpyAsync
// CHECK: %[[RECV_PTR:.*]] = llvm.load %[[RECV_BUF]] : !llvm.ptr -> !llvm.ptr
// CHECK-COUNT-4: llvm.call @cuMemcpyAsync
// CHECK-NOT: llvm.call @nccl
// CHECK: llvm.call @cuStreamSynchronize(%[[STREAM]])
// CHECK-NOT: llvm.call @cuMemcpyAsync
// CHECK: llvm.call @ncclGroupStart() : () -> i32
// CHECK: llvm.call @ncclSend(%[[SEND_PTR]], {{.*}}, %[[STREAM]])
// CHECK: llvm.call @ncclRecv(%[[RECV_PTR]], {{.*}}, %[[STREAM]])
// CHECK: llvm.call @ncclGroupEnd() : () -> i32
// CHECK-NEXT: llvm.return

// CHECK-DAG: llvm.func @ncclGroupStart() -> i32
// CHECK-DAG: llvm.func @ncclGroupEnd() -> i32
// CHECK-DAG: llvm.func @ncclSend(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclRecv(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[BUF:.*]]: tensor<5xi32>, %[[PEER:.*]]: tensor<i32>) -> tensor<5xi32> {
// CHECK: %[[COMM:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: %[[ZEROS:.*]] = stablehlo.constant dense<0> : tensor<5xi32>
// CHECK: %[[RESULT:.*]] = enzymexla.jit_call @enzymexla_jitwrap_ncclGroup_Send_Recv (%[[BUF]], %{{.*}}, %{{.*}}, %[[PEER]], %[[COMM]], %[[ZEROS]], %{{.*}}, %{{.*}}, %[[PEER]], %[[COMM]])
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 5, operand_tuple_indices = []>]
// CHECK-SAME: -> tensor<5xi32>
// CHECK-NEXT: return %[[RESULT]] : tensor<5xi32>
// CHECK-NOT: comm.nccl
