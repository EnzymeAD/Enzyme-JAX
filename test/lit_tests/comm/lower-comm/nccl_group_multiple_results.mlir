// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

// Recv buffers are aliased by operand position, regardless of which members
// come before them.
func.func @main(%buf: tensor<5xi32>, %peer0: tensor<i32>, %peer1: tensor<i32>) -> (tensor<4xf32>, tensor<5xi32>) {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  %results:2 = comm.nccl.group -> tensor<4xf32>, tensor<5xi32> {
    %0 = comm.nccl.recv %peer0, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<4xf32>
    comm.nccl.send %buf, %peer1, %comm : tensor<5xi32>, tensor<i32>, !comm.nccl.comm
    %1 = comm.nccl.recv %peer1, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<5xi32>
    comm.nccl.yield %0, %1 : tensor<4xf32>, tensor<5xi32>
  }
  return %results#0, %results#1 : tensor<4xf32>, tensor<5xi32>
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclGroup_Recv_Send_Recv
// CHECK: llvm.call @ncclGroupStart()
// CHECK: llvm.call @ncclRecv
// CHECK: llvm.call @ncclSend
// CHECK: llvm.call @ncclRecv
// CHECK: llvm.call @ncclGroupEnd()

// CHECK-LABEL: func.func @main
// CHECK: %[[RESULTS:.*]]:2 = enzymexla.jit_call @enzymexla_jitwrap_ncclGroup_Recv_Send_Recv
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 10, operand_tuple_indices = []>]
// CHECK-SAME: -> (tensor<4xf32>, tensor<5xi32>)
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<4xf32>, tensor<5xi32>
