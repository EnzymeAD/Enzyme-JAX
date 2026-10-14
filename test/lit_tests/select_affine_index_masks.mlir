// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns="compare_affine_index_simplify;logical_complementary_half_spaces;select_select_common_branch;select_known_predicate_operand;select_select_same_cond;transpose_symmetric_simplify;transpose_transpose;transpose_simplify;transpose_is_reshape;transpose_elementwise(1);transpose_select;transpose_broadcast_in_dim_to_broadcast_in_dim;add_simplify;and_simplify;div_simplify;select_op_canon(1024)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: stablehlo-translate --interpret --allow-unregistered-dialect %s

// j - i >= 0 holds nowhere... except it holds on and above the diagonal:
// not foldable. i - j + 7 >= 0 holds everywhere on an 8x8 grid.
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

// The upper triangle (j >= i, built as the raising does: iota - broadcast
// iota) or its transpose (i >= j) covers the whole grid.
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

func.func @select_common_true(%p: tensor<4xi1>, %q: tensor<4xi1>, %a: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %inner = stablehlo.select %q, %a, %c : tensor<4xi1>, tensor<4xf32>
  %r = stablehlo.select %p, %a, %inner : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_common_true
// CHECK-NEXT: %[[OR:.+]] = stablehlo.or %arg0, %arg1 : tensor<4xi1>
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %[[OR]], %arg2, %arg3
// CHECK-NEXT: return %[[S]]

func.func @select_common_false(%p: tensor<4xi1>, %q: tensor<4xi1>, %a: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %inner = stablehlo.select %q, %c, %a : tensor<4xi1>, tensor<4xf32>
  %r = stablehlo.select %p, %inner, %a : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_common_false
// CHECK-NEXT: %[[AND:.+]] = stablehlo.and %arg0, %arg1 : tensor<4xi1>
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %[[AND]], %arg3, %arg2
// CHECK-NEXT: return %[[S]]

// Under p, select(p, 0, x) is 0, so (select(p, 0, x) + y) is y.
func.func @select_known_predicate(%p: tensor<4xi1>, %x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
  %z = stablehlo.select %p, %zero, %x : tensor<4xi1>, tensor<4xf32>
  %s = stablehlo.add %z, %y : tensor<4xf32>
  %r = stablehlo.select %p, %s, %x : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_known_predicate
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %arg0, %arg2, %arg1
// CHECK-NEXT: return %[[S]]

// The mirrored-store idiom of a symmetric triangular kernel: the upper
// triangle gets G, the lower triangle gets the transpose of the result.
// With G = X^T X symmetric, that is simply G.
func.func @symmetric_triangular_store(%x: tensor<6x5xf64>, %init: tensor<5x5xf64>) -> tensor<5x5xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<5x5xi64>
  %i1 = stablehlo.iota dim = 0 : tensor<5xi64>
  %ni = stablehlo.negate %i1 : tensor<5xi64>
  %i = stablehlo.iota dim = 0 : tensor<5x5xi64>
  %nj = stablehlo.broadcast_in_dim %ni, dims = [1] : (tensor<5xi64>) -> tensor<5x5xi64>
  %d = stablehlo.add %i, %nj : tensor<5x5xi64>
  %lower = stablehlo.compare GE, %d, %c0 : (tensor<5x5xi64>, tensor<5x5xi64>) -> tensor<5x5xi1>
  %upper = stablehlo.transpose %lower, dims = [1, 0] : (tensor<5x5xi1>) -> tensor<5x5xi1>
  %zero = stablehlo.constant dense<0.0> : tensor<5x5xf64>
  %z = stablehlo.select %upper, %zero, %init : tensor<5x5xi1>, tensor<5x5xf64>
  %g = stablehlo.dot_general %x, %x, contracting_dims = [0] x [0] : (tensor<6x5xf64>, tensor<6x5xf64>) -> tensor<5x5xf64>
  %acc = stablehlo.add %z, %g : tensor<5x5xf64>
  %u = stablehlo.select %upper, %acc, %init : tensor<5x5xi1>, tensor<5x5xf64>
  %ut = stablehlo.transpose %u, dims = [1, 0] : (tensor<5x5xf64>) -> tensor<5x5xf64>
  %u2 = stablehlo.select %upper, %u, %init : tensor<5x5xi1>, tensor<5x5xf64>
  %r = stablehlo.select %lower, %ut, %u2 : tensor<5x5xi1>, tensor<5x5xf64>
  return %r : tensor<5x5xf64>
}
// CHECK-LABEL: func.func @symmetric_triangular_store
// CHECK-NEXT: %[[G:.+]] = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0]
// CHECK-NEXT: return %[[G]]

func.func @main() {
  %x = stablehlo.constant dense<[[1.0, 2.0, 0.5, -1.0, 3.0], [0.0, 1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.5, 2.5, -0.5], [1.0, 1.0, 1.0, 1.0, 1.0], [2.0, -3.0, 4.0, -5.0, 6.0], [0.1, 0.2, 0.3, 0.4, 0.5]]> : tensor<6x5xf64>
  %init = stablehlo.constant dense<7.0> : tensor<5x5xf64>
  %r = func.call @symmetric_triangular_store(%x, %init) : (tensor<6x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>
  %g = stablehlo.dot_general %x, %x, contracting_dims = [0] x [0] : (tensor<6x5xf64>, tensor<6x5xf64>) -> tensor<5x5xf64>
  check.expect_almost_eq %r, %g : tensor<5x5xf64>

  %p = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %q = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %c = stablehlo.constant dense<[-1.0, -2.0, -3.0, -4.0]> : tensor<4xf32>
  %r1 = func.call @select_common_true(%p, %q, %a, %c) : (tensor<4xi1>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e1 = stablehlo.constant dense<[1.0, 2.0, 3.0, -4.0]> : tensor<4xf32>
  check.expect_eq %r1, %e1 : tensor<4xf32>
  %r2 = func.call @select_common_false(%p, %q, %a, %c) : (tensor<4xi1>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e2 = stablehlo.constant dense<[-1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  check.expect_eq %r2, %e2 : tensor<4xf32>
  %r3 = func.call @select_known_predicate(%p, %a, %c) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e3 = stablehlo.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  check.expect_eq %r3, %e3 : tensor<4xf32>

  %m1 = func.call @compare_everywhere() : () -> tensor<8x8xi1>
  %t = stablehlo.constant dense<true> : tensor<8x8xi1>
  check.expect_eq %m1, %t : tensor<8x8xi1>
  %m2 = func.call @compare_nowhere() : () -> tensor<8x8xi1>
  %f = stablehlo.constant dense<false> : tensor<8x8xi1>
  check.expect_eq %m2, %f : tensor<8x8xi1>
  %m3 = func.call @or_covers() : () -> tensor<8x8xi1>
  check.expect_eq %m3, %t : tensor<8x8xi1>
  %m4 = func.call @and_disjoint() : () -> tensor<8x8xi1>
  check.expect_eq %m4, %f : tensor<8x8xi1>
  return
}
