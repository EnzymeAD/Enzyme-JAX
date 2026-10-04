// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu blas_int_width=64" %s | FileCheck %s

// BLAS reads row-major A as A^T, so the triangle always flips.

// dtrsm('L', 'L', 'N', 'N', m = 8, n = 1, lda = 8, ldb = 8)
func.func @main(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<8x1xf64>) -> tensor<8x1xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<transpose>} : (tensor<f64>, tensor<8x8xf64>, tensor<8x1xf64>) -> tensor<8x1xf64>
  return %0 : tensor<8x1xf64>
}

// dtrsm('L', 'U', 'N', 'N', m = 8, n = 3, lda = 8, ldb = 8)
func.func @right(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<3x8xf64>) -> tensor<3x8xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<L>, transa = #enzymexla.transpose<none>} : (tensor<f64>, tensor<8x8xf64>, tensor<3x8xf64>) -> tensor<3x8xf64>
  return %0 : tensor<3x8xf64>
}

// dtrsm('L', 'U', 'T', 'N', m = 8, n = 2, lda = 8, ldb = 8)
func.func @left2(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<8x2xf64>) -> tensor<8x2xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<L>, transa = #enzymexla.transpose<none>} : (tensor<f64>, tensor<8x8xf64>, tensor<8x2xf64>) -> tensor<8x2xf64>
  return %0 : tensor<8x2xf64>
}

// dtrsm('R', 'L', 'N', 'N', m = 16, n = 8, lda = 8, ldb = 16)
func.func @left16(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<8x16xf64>) -> tensor<8x16xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<none>} : (tensor<f64>, tensor<8x8xf64>, tensor<8x16xf64>) -> tensor<8x16xf64>
  return %0 : tensor<8x16xf64>
}

// dtrsm('R', 'L', 'N', 'N', m = 16, n = 8, lda = 8, ldb = 16)
func.func @right16(%alpha: tensor<f64>, %a: tensor<8x8xf64>, %b: tensor<16x8xf64>) -> tensor<16x8xf64> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<transpose>} : (tensor<f64>, tensor<8x8xf64>, tensor<16x8xf64>) -> tensor<16x8xf64>
  return %0 : tensor<16x8xf64>
}

// CHECK: llvm.func @enzymexla_wrapper_blas_dtrsm_
// CHECK: llvm.call @enzymexla_blas_dtrsm_
// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<8x8xf64>, %[[B:.+]]: tensor<8x1xf64>)
// CHECK-NEXT:    %[[L:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CHECK-NEXT:    %[[N:.+]] = stablehlo.constant dense<78> : tensor<i8>
// CHECK-NEXT:    %[[EIGHT:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[ONE:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_ (%[[L]], %[[L]], %[[N]], %[[N]], %[[EIGHT]], %[[ONE]], %[[ALPHA]], %[[A]], %[[EIGHT]], %[[B]], %[[EIGHT]])
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

// CHECK-LABEL: func.func @left2(
// CHECK-SAME:      %[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<8x8xf64>, %[[B:.+]]: tensor<8x2xf64>)
// CHECK-NEXT:    %[[SIDE:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CHECK-NEXT:    %[[UPLO:.+]] = stablehlo.constant dense<85> : tensor<i8>
// CHECK-NEXT:    %[[TRANS:.+]] = stablehlo.constant dense<84> : tensor<i8>
// CHECK-NEXT:    %[[DIAG:.+]] = stablehlo.constant dense<78> : tensor<i8>
// CHECK-NEXT:    %[[EIGHT:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[TWO:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_ (%[[SIDE]], %[[UPLO]], %[[TRANS]], %[[DIAG]], %[[EIGHT]], %[[TWO]], %[[ALPHA]], %[[A]], %[[EIGHT]], %[[B]], %[[EIGHT]])
// CHECK-SAME:      operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]
// CHECK-SAME:      result_layouts = [dense<[0, 1]> : tensor<2xindex>]
// CHECK-NEXT:    return %[[RES]] : tensor<8x2xf64>

// CHECK-LABEL: func.func @left16(
// CHECK-SAME:      %[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<8x8xf64>, %[[B:.+]]: tensor<8x16xf64>)
// CHECK-NEXT:    %[[SIDE:.+]] = stablehlo.constant dense<82> : tensor<i8>
// CHECK-NEXT:    %[[UPLO:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CHECK-NEXT:    %[[NT:.+]] = stablehlo.constant dense<78> : tensor<i8>
// CHECK-NEXT:    %[[SIXTEEN:.+]] = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %[[EIGHT:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_ (%[[SIDE]], %[[UPLO]], %[[NT]], %[[NT]], %[[SIXTEEN]], %[[EIGHT]], %[[ALPHA]], %[[A]], %[[EIGHT]], %[[B]], %[[SIXTEEN]])
// CHECK-SAME:      operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>]
// CHECK-SAME:      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CHECK-NEXT:    return %[[RES]] : tensor<8x16xf64>

// CHECK-LABEL: func.func @right16(
// CHECK-SAME:      %[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<8x8xf64>, %[[B:.+]]: tensor<16x8xf64>)
// CHECK-NEXT:    %[[SIDE:.+]] = stablehlo.constant dense<82> : tensor<i8>
// CHECK-NEXT:    %[[UPLO:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CHECK-NEXT:    %[[NT:.+]] = stablehlo.constant dense<78> : tensor<i8>
// CHECK-NEXT:    %[[SIXTEEN:.+]] = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %[[EIGHT:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = enzymexla.jit_call @enzymexla_wrapper_blas_dtrsm_ (%[[SIDE]], %[[UPLO]], %[[NT]], %[[NT]], %[[SIXTEEN]], %[[EIGHT]], %[[ALPHA]], %[[A]], %[[EIGHT]], %[[B]], %[[SIXTEEN]])
// CHECK-SAME:      operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]
// CHECK-SAME:      result_layouts = [dense<[0, 1]> : tensor<2xindex>]
// CHECK-NEXT:    return %[[RES]] : tensor<16x8xf64>
