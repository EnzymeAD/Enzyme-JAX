// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit{strategy=dependencies})" %s | FileCheck %s

// Lower and fuse Irecv/Wait and Isend/Wait. Keep the add between them:
// the send needs its result.

module {
  func.func @computed_send(%recvbuf: tensor<4xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %delta: tensor<4xi32>) -> tensor<4xi32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %count = stablehlo.constant dense<4> : tensor<i32>
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %recv_tag = stablehlo.constant dense<10> : tensor<i32>
    %send_tag = stablehlo.constant dense<20> : tensor<i32>
    %outbuf, %request = enzymexla.mpi.irecv(%recvbuf, %count, %peer, %recv_tag) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
    enzymexla.mpi.wait(%request) : tensor<i32>
    %computed = stablehlo.add %outbuf, %delta : tensor<4xi32>
    %send_request = enzymexla.mpi.isend(%computed, %count, %peer, %send_tag) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    enzymexla.mpi.wait(%send_request) : tensor<i32>
    return %computed : tensor<4xi32>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait(
// CHECK-SAME: %[[RECV_BUFFER:[^ :]+]]: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %[[RECV_REQUEST:[^ :]+]]: !llvm.ptr) {
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Irecv(%[[RECV_BUFFER]], {{.*}}, %[[RECV_REQUEST]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[RECV_REQUEST]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Wait(
// CHECK-SAME: %[[SEND_BUFFER:[^ :]+]]: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %[[SEND_REQUEST:[^ :]+]]: !llvm.ptr) {
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Isend(%[[SEND_BUFFER]], {{.*}}, %[[SEND_REQUEST]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[SEND_REQUEST]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @computed_send(
// CHECK-SAME: %[[BUFFER:[^ :]+]]: tensor<4xi32>{{.*}}, %[[DELTA:[^ :]+]]: tensor<4xi32>)
// CHECK-NOT: enzymexla.jit_call
// CHECK: %[[RECV:.*]]:2 = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait (%[[BUFFER]], %[[COUNT:[^ ,]+]], %[[PEER:[^ ,]+]], %{{[^ ,]+}}, %{{[^ ,)]+}})
// CHECK-SAME: {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>]} : {{.*}} -> (tensor<4xi32>, tensor<i32>)
// CHECK-NEXT: %[[COMPUTED:.*]] = stablehlo.add %[[RECV]]#0, %[[DELTA]] : tensor<4xi32>
// CHECK: %{{[^ ]+}} = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Wait (%[[COMPUTED]], %[[COUNT]], %[[PEER]], %{{[^ ,]+}}, %{{[^ ,)]+}})
// CHECK-SAME: {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 4, operand_tuple_indices = []>]} : {{.*}} -> tensor<i32>
// CHECK-NEXT: return %[[COMPUTED]] : tensor<4xi32>
// CHECK-NEXT: }

