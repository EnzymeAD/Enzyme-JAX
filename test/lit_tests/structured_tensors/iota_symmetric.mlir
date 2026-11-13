// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Reactant Symmetric(A, :U): select(iota(0) < iota(1), A, transpose(A, [1, 0])) (#1609)
// CHECK-LABEL: func.func @reactant_symmetric_upper
// CHECK-SAME: (%[[A:.+]]: tensor<4x4xf32>)
// CHECK: %[[SYM:.+]] = stablehlo.select %{{.+}}, %[[A]], %{{.+}} {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK-NEXT: return %[[SYM]] : tensor<4x4xf32>
func.func @reactant_symmetric_upper(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = stablehlo.iota dim = 0 : tensor<4x4xi64>
  %c = stablehlo.iota dim = 1 : tensor<4x4xi64>
  %cmp = stablehlo.compare LT, %r, %c : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
  %at = stablehlo.transpose %a, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %sym = stablehlo.select %cmp, %a, %at : tensor<4x4xi1>, tensor<4x4xf32>
  %res = stablehlo.transpose %sym, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}

// Reactant Symmetric(A, :L): select(iota(0) > iota(1), A, transpose(A, [1, 0]))
// CHECK-LABEL: func.func @reactant_symmetric_lower
// CHECK-SAME: (%[[A:.+]]: tensor<4x4xf32>)
// CHECK: %[[SYM:.+]] = stablehlo.select %{{.+}}, %[[A]], %{{.+}} {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK-NEXT: return %[[SYM]] : tensor<4x4xf32>
func.func @reactant_symmetric_lower(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = stablehlo.iota dim = 0 : tensor<4x4xi64>
  %c = stablehlo.iota dim = 1 : tensor<4x4xi64>
  %cmp = stablehlo.compare GT, %r, %c : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
  %at = stablehlo.transpose %a, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %sym = stablehlo.select %cmp, %a, %at : tensor<4x4xi1>, tensor<4x4xf32>
  %res = stablehlo.transpose %sym, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}

// Diagonal matrix via iota EQ with asymmetric broadcast onTrue and zero onFalse
// CHECK-LABEL: func.func @iota_eq_diagonal_symmetric
// CHECK-SAME: (%[[D:.+]]: tensor<4xf32>)
// CHECK: %[[SYM:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK-NEXT: return %[[SYM]] : tensor<4x4xf32>
func.func @iota_eq_diagonal_symmetric(%d: tensor<4xf32>) -> tensor<4x4xf32> {
  %r = stablehlo.iota dim = 0 : tensor<4x4xi64>
  %c = stablehlo.iota dim = 1 : tensor<4x4xi64>
  %cmp = stablehlo.compare EQ, %r, %c : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
  %db = stablehlo.broadcast_in_dim %d, dims = [0] : (tensor<4xf32>) -> tensor<4x4xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %diag = stablehlo.select %cmp, %db, %zero : tensor<4x4xi1>, tensor<4x4xf32>
  %res = stablehlo.transpose %diag, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}

// Negative test: select(iota(0) < iota(1), A, B) with unrelated B is NOT symmetric
// CHECK-LABEL: func.func @iota_lt_unrelated_not_symmetric
// CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %arg0, %arg1 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK-NEXT: %[[TR:.+]] = stablehlo.transpose %[[SEL]], dims = [1, 0]
// CHECK-NEXT: return %[[TR]] : tensor<4x4xf32>
func.func @iota_lt_unrelated_not_symmetric(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = stablehlo.iota dim = 0 : tensor<4x4xi64>
  %c = stablehlo.iota dim = 1 : tensor<4x4xi64>
  %cmp = stablehlo.compare LT, %r, %c : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
  %sel = stablehlo.select %cmp, %a, %b : tensor<4x4xi1>, tensor<4x4xf32>
  %res = stablehlo.transpose %sel, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}
