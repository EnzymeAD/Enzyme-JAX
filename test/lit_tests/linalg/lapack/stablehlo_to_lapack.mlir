// RUN: enzymexlamlir-opt --stablehlo-to-lapack %s | FileCheck %s

// potrf leaves the input in the other triangle, hence the triangle mask.

func.func @upper(%a: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = stablehlo.cholesky %a : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}
// CHECK-LABEL: func.func @upper
// CHECK-SAME:      (%[[A:.+]]: tensor<8x8xf64>) -> tensor<8x8xf64>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK-NEXT:    %[[OUT:.+]], %[[INFO:.+]] = enzymexla.lapack.potrf %[[A]] {uplo = #enzymexla.uplo<U>} : (tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<i64>)
// CHECK-NEXT:    %[[ROWS:.+]] = stablehlo.iota dim = 0 : tensor<8x8xi64>
// CHECK-NEXT:    %[[COLS:.+]] = stablehlo.iota dim = 1 : tensor<8x8xi64>
// CHECK-NEXT:    %[[KEEP:.+]] = stablehlo.compare LE, %[[ROWS]], %[[COLS]] : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.select %[[KEEP]], %[[OUT]], %[[ZERO]] : tensor<8x8xi1>, tensor<8x8xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<8x8xf64>

func.func @lower(%a: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = stablehlo.cholesky %a, lower = true : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}
// CHECK-LABEL: func.func @lower
// CHECK:         enzymexla.lapack.potrf %{{.+}} {uplo = #enzymexla.uplo<L>}
// CHECK:         stablehlo.compare GE,

func.func @batched(%a: tensor<4x8x8xf64>) -> tensor<4x8x8xf64> {
  %0 = stablehlo.cholesky %a : tensor<4x8x8xf64>
  return %0 : tensor<4x8x8xf64>
}
// CHECK-LABEL: func.func @batched
// CHECK-NEXT:    %[[B:.+]] = stablehlo.cholesky %{{.+}} : tensor<4x8x8xf64>
// CHECK-NEXT:    return %[[B]]

func.func @solve(%a: tensor<8x8xf64>, %b: tensor<8x1xf64>) -> tensor<8x1xf64> {
  %0 = "stablehlo.triangular_solve"(%a, %b) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<8x8xf64>, tensor<8x1xf64>) -> tensor<8x1xf64>
  return %0 : tensor<8x1xf64>
}
// CHECK-LABEL: func.func @solve
// CHECK-SAME:      (%[[SA:.+]]: tensor<8x8xf64>, %[[SB:.+]]: tensor<8x1xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:    %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[X:.+]] = enzymexla.blas.trsm %[[ONE]], %[[SA]], %[[SB]] {side = #enzymexla.side<left>, transa = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<U>} : (tensor<f64>, tensor<8x8xf64>, tensor<8x1xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:    return %[[X]] : tensor<8x1xf64>
