// RUN: enzymexlamlir-opt --lower-enzymexla-lapack={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) {
  %0:3 = enzymexla.lapack.syevd %arg0 {uplo = #enzymexla.uplo<L>} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>)
  return %0#0, %0#1, %0#2 : tensor<4x4xf64>, tensor<4xf64>, tensor<i64>
}

// CPU-LABEL:  llvm.func @enzymexla_wrapper_lapack_dsyevd_
// CPU-LABEL:  llvm.func @enzymexla_lapack_dsyevd_

// CPU:  func.func @main(%[[ARG:.*]]: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) {
// CPU-NEXT:    %[[W:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CPU-NEXT:    %[[INFO:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %[[RES:.*]]:3 = enzymexla.jit_call @enzymexla_wrapper_lapack_dsyevd_ (%[[ARG]], %[[W]], %[[INFO]])
// CPU-SAME:      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]
// CPU-SAME:      xla_side_effect_free
// CPU-SAME:      (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>)
// CPU-NEXT:    return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : tensor<4x4xf64>, tensor<4xf64>, tensor<i64>
// CPU-NEXT:  }
