// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu blas_int_width=64" %s | FileCheck %s

// Unbatched f64 trsm on CPU lowers to a direct BLAS dtrsm jit_call
// (column-major layouts; B aliased with the solution).

func.func @main(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<8x1xf64>) -> tensor<8x1xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<transpose>} : (tensor<f64>, tensor<8x8xf64>, tensor<8x1xf64>) -> tensor<8x1xf64>
  return %0 : tensor<8x1xf64>
}
// CHECK: llvm.func @enzymexla_wrapper_blas_dtrsm_
// CHECK: llvm.call @enzymexla_blas_dtrsm_
// CHECK-LABEL: func.func @main
// CHECK: %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>]
// CHECK: return %[[RES]] : tensor<8x1xf64>
