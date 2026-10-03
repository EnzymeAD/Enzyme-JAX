// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu blas_int_width=64" %s | FileCheck %s

// Unbatched f64 trsm on CPU lowers to a direct BLAS dtrsm jit_call on the
// row-major buffers (layouts [1, 0], B aliased with the solution). BLAS reads
// them column-major, i.e. transposed, so op(A) X = B is passed as
// X^T op(A^T) = B^T: the side and the triangle are flipped, the transpose flag
// is kept, and m, n and ldb describe B^T.

// left, upper, transpose on an 8 x 1 RHS -> dtrsm('R', 'L', 'T', 'N', m = 1, n = 8, lda = 8, ldb = 1)
func.func @main(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<8x1xf64>) -> tensor<8x1xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<transpose>} : (tensor<f64>, tensor<8x8xf64>, tensor<8x1xf64>) -> tensor<8x1xf64>
  return %0 : tensor<8x1xf64>
}

// right, lower, no transpose on a 3 x 8 RHS -> dtrsm('L', 'U', 'N', 'N', m = 8, n = 3, lda = 8, ldb = 8)
func.func @right(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<3x8xf64>) -> tensor<3x8xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<L>, transa = #enzymexla.transpose<none>} : (tensor<f64>, tensor<8x8xf64>, tensor<3x8xf64>) -> tensor<3x8xf64>
  return %0 : tensor<3x8xf64>
}

// CHECK: llvm.func @enzymexla_wrapper_blas_dtrsm_
// CHECK: llvm.call @enzymexla_blas_dtrsm_
// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<8x8xf64>, %[[B:.+]]: tensor<8x1xf64>)
// CHECK-NEXT:    %[[SIDE:.+]] = stablehlo.constant dense<82> : tensor<i8>
// CHECK-NEXT:    %[[UPLO:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CHECK-NEXT:    %[[TRANS:.+]] = stablehlo.constant dense<84> : tensor<i8>
// CHECK-NEXT:    %[[DIAG:.+]] = stablehlo.constant dense<78> : tensor<i8>
// CHECK-NEXT:    %[[ONE:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[EIGHT:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_ (%[[SIDE]], %[[UPLO]], %[[TRANS]], %[[DIAG]], %[[ONE]], %[[EIGHT]], %[[ALPHA]], %[[A]], %[[EIGHT]], %[[B]], %[[ONE]])
// CHECK-SAME:      operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>]
// CHECK-SAME:      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>]
// CHECK-SAME:      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CHECK-NEXT:    return %[[RES]] : tensor<8x1xf64>

// CHECK-LABEL: func.func @right(
// CHECK-SAME:      %[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<8x8xf64>, %[[B:.+]]: tensor<3x8xf64>)
// CHECK-NEXT:    %[[SIDE:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CHECK-NEXT:    %[[UPLO:.+]] = stablehlo.constant dense<85> : tensor<i8>
// CHECK-NEXT:    %[[NT:.+]] = stablehlo.constant dense<78> : tensor<i8>
// CHECK-NEXT:    %[[EIGHT:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[THREE:.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_ (%[[SIDE]], %[[UPLO]], %[[NT]], %[[NT]], %[[EIGHT]], %[[THREE]], %[[ALPHA]], %[[A]], %[[EIGHT]], %[[B]], %[[EIGHT]])
// CHECK-SAME:      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CHECK-NEXT:    return %[[RES]] : tensor<3x8xf64>
