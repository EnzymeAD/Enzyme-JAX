// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

// Lower five independent MPI calls, then fuse them into one JIT while
// preserving call order, request buffers, and output aliases.

module {
  func.func @mpi_order(%sendbuf: tensor<5xi32>, %recvbuf: tensor<5xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}, %control: tensor<1xi32>) -> tensor<5xi32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %count = stablehlo.constant dense<5> : tensor<i32>
    %tag_a = stablehlo.constant dense<10> : tensor<i32>
    %tag_b = stablehlo.constant dense<20> : tensor<i32>
    %tag_control = stablehlo.constant dense<30> : tensor<i32>
    %send_request = enzymexla.mpi.isend(%sendbuf, %count, %peer, %tag_a) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %outbuf, %recv_request = enzymexla.mpi.irecv(%recvbuf, %count, %peer, %tag_b) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xi32>, tensor<i32>)
    enzymexla.mpi.wait(%send_request) : tensor<i32>
    // Reuse peer=1 as the control message count (one element) and destination.
    enzymexla.mpi.send(%control, %peer, %peer, %tag_control) {datatype = #enzymexla.datatype<MPI_INT>} : tensor<1xi32>, tensor<i32>, tensor<i32>, tensor<i32>
    enzymexla.mpi.wait(%recv_request) : tensor<i32>
    return %outbuf : tensor<5xi32>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait_enzymexla_wrapper_MPI_Send_MPI_INT_enzymexla_wrapper_MPI_Wait(
// CHECK-SAME: %[[SEND_BUF:[^ :]+]]: !llvm.ptr, %[[COUNT_PTR:[^ :]+]]: !llvm.ptr, %[[PEER_PTR:[^ :]+]]: !llvm.ptr, %[[TAG_A_PTR:[^ :]+]]: !llvm.ptr, %[[SEND_SLOT:[^ :]+]]: !llvm.ptr, %[[RECV_BUF:[^ :]+]]: !llvm.ptr, %[[TAG_B_PTR:[^ :]+]]: !llvm.ptr, %[[RECV_SLOT:[^ :]+]]: !llvm.ptr, %[[CONTROL_BUF:[^ :]+]]: !llvm.ptr, %[[CONTROL_TAG_PTR:[^ :]+]]: !llvm.ptr) {
// CHECK-NOT: llvm.call
// CHECK: %[[SEND_COUNT:.*]] = llvm.load %[[COUNT_PTR]]
// CHECK-NEXT: %[[SEND_PEER:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[SEND_TAG:.*]] = llvm.load %[[TAG_A_PTR]]
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Isend(%[[SEND_BUF]], %[[SEND_COUNT]], %{{[^ ,]+}}, %[[SEND_PEER]], %[[SEND_TAG]], %{{[^ ,]+}}, %[[SEND_SLOT]])
// CHECK-NOT: llvm.call
// CHECK: %[[RECV_COUNT:.*]] = llvm.load %[[COUNT_PTR]]
// CHECK-NEXT: %[[RECV_PEER:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[RECV_TAG:.*]] = llvm.load %[[TAG_B_PTR]]
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Irecv(%[[RECV_BUF]], %[[RECV_COUNT]], %{{[^ ,]+}}, %[[RECV_PEER]], %[[RECV_TAG]], %{{[^ ,]+}}, %[[RECV_SLOT]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[SEND_SLOT]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: %[[CONTROL_COUNT:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[CONTROL_PEER:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[CONTROL_TAG:.*]] = llvm.load %[[CONTROL_TAG_PTR]]
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Send(%[[CONTROL_BUF]], %[[CONTROL_COUNT]], %{{[^ ,]+}}, %[[CONTROL_PEER]], %[[CONTROL_TAG]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[RECV_SLOT]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @mpi_order(
// CHECK-SAME: %[[SEND:[^ :]+]]: tensor<5xi32>, %[[RECV:[^ :]+]]: tensor<5xi32>{{.*}}, %[[CONTROL:[^ :]+]]: tensor<1xi32>)
// CHECK-NOT: enzymexla.jit_call
// CHECK-DAG: %[[PEER:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[COUNT:.*]] = stablehlo.constant dense<5> : tensor<i32>
// CHECK-DAG: %[[TAG_A:.*]] = stablehlo.constant dense<10> : tensor<i32>
// CHECK-DAG: %[[TAG_B:.*]] = stablehlo.constant dense<20> : tensor<i32>
// CHECK-DAG: %[[CONTROL_TAG:.*]] = stablehlo.constant dense<30> : tensor<i32>
// CHECK-NOT: enzymexla.jit_call
// CHECK: %[[MPI:.*]]:3 = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait_enzymexla_wrapper_MPI_Send_MPI_INT_enzymexla_wrapper_MPI_Wait (%[[SEND]], %[[COUNT]], %[[PEER]], %[[TAG_A]], %{{[^ ,]+}}, %[[RECV]], %[[TAG_B]], %{{[^ ,]+}}, %[[CONTROL]], %[[CONTROL_TAG]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>]} : {{.*}} -> (tensor<5xi32>, tensor<i32>, tensor<i32>)
// CHECK-NEXT: return %[[MPI]]#0 : tensor<5xi32>
// CHECK-NEXT: }
