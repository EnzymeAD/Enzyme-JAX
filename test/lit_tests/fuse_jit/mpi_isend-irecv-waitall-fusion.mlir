// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

// A mixed nonblocking component must fuse as one unit while retaining lexical
// Isend -> Irecv -> Waitall order in the generated wrapper.

module {
  func.func @main(%sendbuf: tensor<5xi32>, %recvbuf: tensor<5xi32>) -> tensor<5xi32> {
    %count = stablehlo.constant dense<5> : tensor<i32>
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %send_tag = stablehlo.constant dense<41> : tensor<i32>
    %recv_tag = stablehlo.constant dense<42> : tensor<i32>
    %send_request = enzymexla.mpi.isend(%sendbuf, %count, %peer, %send_tag) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %recv_ready, %recv_request = enzymexla.mpi.irecv(%recvbuf, %count, %peer, %recv_tag) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xi32>, tensor<i32>)
    enzymexla.mpi.waitall(%send_request, %recv_request) : tensor<i32>, tensor<i32>
    return %recv_ready : tensor<5xi32>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Waitall_2
// CHECK:         llvm.call @MPI_Isend({{.*}}, %[[SEND_REQUEST:[^ ,)]+]])
// CHECK:         llvm.call @MPI_Irecv({{.*}}, %[[RECV_REQUEST:[^ ,)]+]])
// CHECK:         %[[SEND_REQUEST_VALUE:.*]] = llvm.load %[[SEND_REQUEST]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[SEND_REQUEST_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         %[[RECV_REQUEST_VALUE:.*]] = llvm.load %[[RECV_REQUEST]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[RECV_REQUEST_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Waitall
// CHECK:         llvm.return

// CHECK-LABEL: func.func @main
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK:         %[[FUSED:.*]] = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Waitall_2
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 5, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall_2
// CHECK:         return %[[FUSED]] : tensor<5xi32>
