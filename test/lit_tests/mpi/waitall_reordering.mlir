// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%buf: tensor<4xf32>) -> tensor<4xf32> {
    %request = stablehlo.constant dense<-1> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %request, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    // This now parses correctly and creates the SSA edge
    %buf_ready = enzymexla.mpi.waitall(%c_1, %5, %buf) : (tensor<i32>, tensor<1xi32>, tensor<4xf32>) -> tensor<4xf32>
    return %buf_ready : tensor<4xf32>
  }
}

// CPU: enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall_with_inbuf
// CPU-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]
