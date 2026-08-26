// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

// Two independent communication chains must remain separate fusion components.

module {
  func.func @main(%recvbuf0: tensor<4xi32>, %recvbuf1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %count = stablehlo.constant dense<4> : tensor<i32>
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %tag0 = stablehlo.constant dense<20> : tensor<i32>
    %tag1 = stablehlo.constant dense<21> : tensor<i32>
    %recv_ready0, %request0 = enzymexla.mpi.irecv(%recvbuf0, %count, %peer, %tag0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
    enzymexla.mpi.wait(%request0) : tensor<i32>
    %recv_ready1, %request1 = enzymexla.mpi.irecv(%recvbuf1, %count, %peer, %tag1) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
    enzymexla.mpi.wait(%request1) : tensor<i32>
    return %recv_ready0, %recv_ready1 : tensor<4xi32>, tensor<4xi32>
  }
}

// CHECK-NOT: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait_enzymexla_wrapper_MPI_Irecv_MPI_INT

// CHECK-LABEL: func.func @main
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK:         %[[FIRST:.*]] = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CHECK:         %[[SECOND:.*]] = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait_1
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Wait
// CHECK:         return %[[FIRST]], %[[SECOND]] : tensor<4xi32>, tensor<4xi32>
