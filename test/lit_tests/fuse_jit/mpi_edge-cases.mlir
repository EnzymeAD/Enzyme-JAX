// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

module {
  // A side-effecting component remains observable even when it has no results.
  func.func @isend_wait_resultless(%sendbuf: tensor<4xi32>) {
    %count = stablehlo.constant dense<4> : tensor<i32>
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %tag = stablehlo.constant dense<10> : tensor<i32>
    %request = enzymexla.mpi.isend(%sendbuf, %count, %peer, %tag) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    enzymexla.mpi.wait(%request) : tensor<i32>
    return
  }

  // Both Irecv results escape the component: the request is consumed by Wait
  // and is also returned by the enclosing function.
  func.func @escaping_intermediate(%recvbuf: tensor<4xi32>) -> (tensor<4xi32>, tensor<i32>) {
    %count = stablehlo.constant dense<4> : tensor<i32>
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %tag = stablehlo.constant dense<11> : tensor<i32>
    %recv_ready, %request = enzymexla.mpi.irecv(%recvbuf, %count, %peer, %tag) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
    enzymexla.mpi.wait(%request) : tensor<i32>
    return %recv_ready, %request : tensor<4xi32>, tensor<i32>
  }

  // Zero-argument calls have no SSA edge, so adjacency alone cannot fuse them.
  func.func @barriers_no_ssa_edge() {
    enzymexla.mpi.barrier
    enzymexla.mpi.barrier
    return
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Wait
// CHECK:         llvm.call @MPI_Isend
// CHECK:         llvm.call @MPI_Wait
// CHECK:         llvm.return

// CHECK-LABEL: func.func @isend_wait_resultless
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT
// CHECK:         enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Wait
// CHECK-SAME:    -> ()
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Wait
// CHECK:         return

// CHECK-LABEL: func.func @escaping_intermediate
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK:         %[[FUSED:.*]]:2 = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Wait
// CHECK:         return %[[FUSED]]#0, %[[FUSED]]#1 : tensor<4xi32>, tensor<i32>

// CHECK-LABEL: func.func @barriers_no_ssa_edge
// CHECK-NOT:     enzymexla.jit_call @fused__
// CHECK:         enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
// CHECK-NEXT:    enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
// CHECK-NEXT:    return
