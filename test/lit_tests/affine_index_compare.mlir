// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns="compare_affine_index_simplify;logical_complementary_half_spaces" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: stablehlo-translate --interpret --allow-unregistered-dialect %s

// i - j + 7 >= 0 holds everywhere on an 8x8 grid.
func.func @compare_everywhere() -> tensor<8x8xi1> {
  %c7 = stablehlo.constant dense<7> : tensor<8x8xi64>
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %j = stablehlo.iota dim = 1 : tensor<8x8xi64>
  %d = stablehlo.subtract %i, %j : tensor<8x8xi64>
  %e = stablehlo.add %d, %c7 : tensor<8x8xi64>
  %c0 = stablehlo.constant dense<0> : tensor<8x8xi64>
  %m = stablehlo.compare GE, %e, %c0 : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  return %m : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @compare_everywhere
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<true> : tensor<8x8xi1>
// CHECK-NEXT: return %[[C]]

// i - j > 7 never holds on an 8x8 grid.
func.func @compare_nowhere() -> tensor<8x8xi1> {
  %c7 = stablehlo.constant dense<7> : tensor<8x8xi64>
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %j = stablehlo.iota dim = 1 : tensor<8x8xi64>
  %d = stablehlo.subtract %i, %j : tensor<8x8xi64>
  %m = stablehlo.compare GT, %d, %c7 : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  return %m : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @compare_nowhere
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<false> : tensor<8x8xi1>
// CHECK-NEXT: return %[[C]]

// A genuinely triangular mask is left alone.
func.func @compare_triangular() -> tensor<8x8xi1> {
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %j = stablehlo.iota dim = 1 : tensor<8x8xi64>
  %m = stablehlo.compare GE, %j, %i : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  return %m : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @compare_triangular
// CHECK: stablehlo.compare GE

// A comparison against itself always holds.
func.func @compare_self() -> tensor<1520xi1> {
  %0 = stablehlo.iota dim = 0 : tensor<1520xi64>
  %1 = stablehlo.compare GE, %0, %0 : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1>
  return %1 : tensor<1520xi1>
}
// CHECK-LABEL: func.func @compare_self
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<true> : tensor<1520xi1>
// CHECK-NEXT: return %[[C]]

// The lower triangle (i >= j, built as the raising does: iota - broadcast
// iota) or its transpose (j >= i) covers the whole grid.
func.func @or_covers() -> tensor<8x8xi1> {
  %c0 = stablehlo.constant dense<0> : tensor<8x8xi64>
  %i1 = stablehlo.iota dim = 0 : tensor<8xi64>
  %ni = stablehlo.negate %i1 : tensor<8xi64>
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %nj = stablehlo.broadcast_in_dim %ni, dims = [1] : (tensor<8xi64>) -> tensor<8x8xi64>
  %d = stablehlo.add %i, %nj : tensor<8x8xi64>
  %lower = stablehlo.compare GE, %d, %c0 : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %upper = stablehlo.transpose %lower, dims = [1, 0] : (tensor<8x8xi1>) -> tensor<8x8xi1>
  %o = stablehlo.or %lower, %upper : tensor<8x8xi1>
  return %o : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @or_covers
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<true> : tensor<8x8xi1>
// CHECK-NEXT: return %[[C]]

// Strict triangles j > i and i > j leave the diagonal uncovered.
func.func @or_strict_not_covered() -> tensor<8x8xi1> {
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %j = stablehlo.iota dim = 1 : tensor<8x8xi64>
  %a = stablehlo.compare GT, %j, %i : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %b = stablehlo.compare GT, %i, %j : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %o = stablehlo.or %a, %b : tensor<8x8xi1>
  return %o : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @or_strict_not_covered
// CHECK: stablehlo.or

// Strict triangles are disjoint.
func.func @and_disjoint() -> tensor<8x8xi1> {
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %j = stablehlo.iota dim = 1 : tensor<8x8xi64>
  %a = stablehlo.compare GT, %j, %i : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %b = stablehlo.compare GT, %i, %j : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %o = stablehlo.and %a, %b : tensor<8x8xi1>
  return %o : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @and_disjoint
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<false> : tensor<8x8xi1>
// CHECK-NEXT: return %[[C]]

// Non-strict triangles share the diagonal.
func.func @and_shares_diagonal() -> tensor<8x8xi1> {
  %i = stablehlo.iota dim = 0 : tensor<8x8xi64>
  %j = stablehlo.iota dim = 1 : tensor<8x8xi64>
  %a = stablehlo.compare GE, %j, %i : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %b = stablehlo.compare GE, %i, %j : (tensor<8x8xi64>, tensor<8x8xi64>) -> tensor<8x8xi1>
  %o = stablehlo.and %a, %b : tensor<8x8xi1>
  return %o : tensor<8x8xi1>
}
// CHECK-LABEL: func.func @and_shares_diagonal
// CHECK: stablehlo.and

func.func @main() {
  %t = stablehlo.constant dense<true> : tensor<8x8xi1>
  %f = stablehlo.constant dense<false> : tensor<8x8xi1>
  %m1 = func.call @compare_everywhere() : () -> tensor<8x8xi1>
  check.expect_eq %m1, %t : tensor<8x8xi1>
  %m2 = func.call @compare_nowhere() : () -> tensor<8x8xi1>
  check.expect_eq %m2, %f : tensor<8x8xi1>
  %m3 = func.call @or_covers() : () -> tensor<8x8xi1>
  check.expect_eq %m3, %t : tensor<8x8xi1>
  %m4 = func.call @and_disjoint() : () -> tensor<8x8xi1>
  check.expect_eq %m4, %f : tensor<8x8xi1>
  %m5 = func.call @compare_self() : () -> tensor<1520xi1>
  %t5 = stablehlo.constant dense<true> : tensor<1520xi1>
  check.expect_eq %m5, %t5 : tensor<1520xi1>
  return
}
