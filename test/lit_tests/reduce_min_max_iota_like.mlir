// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns="reduce_min_max_iota_like;min_max_iota_const_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: stablehlo-translate --interpret --allow-unregistered-dialect %s

// The reduced progression of https://github.com/EnzymeAD/Enzyme-JAX/issues/3530:
// max over i of max(1200 - i, 0) with a size-1200 iota, which should fold to
// 1200 without materializing the iota.
func.func @issue_3530() -> tensor<i64> {
  %c = stablehlo.constant dense<1200> : tensor<1200xi64>
  %c_2 = stablehlo.constant dense<0> : tensor<i64>
  %c_3 = stablehlo.constant dense<0> : tensor<1200xi64>
  %6 = stablehlo.iota dim = 0 : tensor<1200xi64>
  %7 = stablehlo.subtract %c, %6 : tensor<1200xi64>
  %8 = stablehlo.maximum %7, %c_3 : tensor<1200xi64>
  %9 = stablehlo.reduce(%8 init: %c_2) applies stablehlo.maximum across dimensions = [0] : (tensor<1200xi64>, tensor<i64>) -> tensor<i64>
  return %9 : tensor<i64>
}
// CHECK-LABEL: func.func @issue_3530
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<1200> : tensor<i64>
// CHECK-NEXT: return %[[C]]

// A tensor far too large to constant fold element by element.
func.func @huge_iota_max() -> tensor<i64> {
  %init = stablehlo.constant dense<-1> : tensor<i64>
  %iota = stablehlo.iota dim = 0 : tensor<1000000000000xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<1000000000000xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @huge_iota_max
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<999999999999> : tensor<i64>
// CHECK-NEXT: return %[[C]]

// The init value dominates the whole progression.
func.func @iota_max_init_dominates() -> tensor<i64> {
  %init = stablehlo.constant dense<100> : tensor<i64>
  %iota = stablehlo.iota dim = 0 : tensor<10xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<10xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @iota_max_init_dominates
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<100> : tensor<i64>
// CHECK-NEXT: return %[[C]]

// min of a scaled, offset iota: 3 * i + 5 for i in [0, 7).
func.func @scaled_iota_min() -> tensor<i32> {
  %init = stablehlo.constant dense<2147483647> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<7xi32>
  %c5 = stablehlo.constant dense<5> : tensor<7xi32>
  %iota = stablehlo.iota dim = 0 : tensor<7xi32>
  %0 = stablehlo.multiply %iota, %c3 : tensor<7xi32>
  %1 = stablehlo.add %0, %c5 : tensor<7xi32>
  %2 = stablehlo.reduce(%1 init: %init) applies stablehlo.minimum across dimensions = [0] : (tensor<7xi32>, tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}
// CHECK-LABEL: func.func @scaled_iota_min
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<5> : tensor<i32>
// CHECK-NEXT: return %[[C]]

// Negated iota: min of -i for i in [0, 7) is -6.
func.func @negated_iota_min() -> tensor<i32> {
  %init = stablehlo.constant dense<2147483647> : tensor<i32>
  %iota = stablehlo.iota dim = 0 : tensor<7xi32>
  %0 = stablehlo.negate %iota : tensor<7xi32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.minimum across dimensions = [0] : (tensor<7xi32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}
// CHECK-LABEL: func.func @negated_iota_min
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<-6> : tensor<i32>
// CHECK-NEXT: return %[[C]]

// Reducing both dimensions of a 2D iota, where the iota runs along dim 1.
func.func @iota_2d_full_reduce() -> tensor<i64> {
  %init = stablehlo.constant dense<0> : tensor<i64>
  %iota = stablehlo.iota dim = 1 : tensor<4x9xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<4x9xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @iota_2d_full_reduce
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT: return %[[C]]

// Reducing only the iota dimension of a 2D iota: a splat over the other dim.
func.func @iota_2d_reduce_iota_dim() -> tensor<4xi64> {
  %init = stablehlo.constant dense<0> : tensor<i64>
  %iota = stablehlo.iota dim = 1 : tensor<4x9xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<4x9xi64>, tensor<i64>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @iota_2d_reduce_iota_dim
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<8> : tensor<4xi64>
// CHECK-NEXT: return %[[C]]

// Reducing a dimension along which the input is constant: the iota survives.
func.func @iota_2d_reduce_other_dim() -> tensor<9xi64> {
  %init = stablehlo.constant dense<-5> : tensor<i64>
  %iota = stablehlo.iota dim = 1 : tensor<4x9xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<4x9xi64>, tensor<i64>) -> tensor<9xi64>
  return %0 : tensor<9xi64>
}
// CHECK-LABEL: func.func @iota_2d_reduce_other_dim
// CHECK-NEXT: %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<9xi64>
// CHECK-NEXT: return %[[IOTA]]

// Same, but the init value cuts into the progression: max(i, 3).
func.func @iota_2d_reduce_other_dim_init_clamps() -> tensor<9xi64> {
  %init = stablehlo.constant dense<3> : tensor<i64>
  %iota = stablehlo.iota dim = 1 : tensor<4x9xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<4x9xi64>, tensor<i64>) -> tensor<9xi64>
  return %0 : tensor<9xi64>
}
// CHECK-LABEL: func.func @iota_2d_reduce_other_dim_init_clamps
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<9xi64>
// CHECK: %[[BC:.+]] = stablehlo.broadcast_in_dim %[[INIT]], dims = [] : (tensor<i64>) -> tensor<9xi64>
// CHECK-NEXT: %[[MAX:.+]] = stablehlo.maximum %[[IOTA]], %[[BC]] : tensor<9xi64>
// CHECK-NEXT: return %[[MAX]]

// Reducing a constant dimension of `10 - 2 * i` (iota along dim 0 of 5x3).
func.func @scaled_iota_reduce_other_dim() -> tensor<5xi64> {
  %init = stablehlo.constant dense<-100> : tensor<i64>
  %c10 = stablehlo.constant dense<10> : tensor<5x3xi64>
  %c2 = stablehlo.constant dense<2> : tensor<5x3xi64>
  %iota = stablehlo.iota dim = 0 : tensor<5x3xi64>
  %0 = stablehlo.multiply %iota, %c2 : tensor<5x3xi64>
  %1 = stablehlo.subtract %c10, %0 : tensor<5x3xi64>
  %2 = stablehlo.reduce(%1 init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<5x3xi64>, tensor<i64>) -> tensor<5xi64>
  return %2 : tensor<5xi64>
}
// CHECK-LABEL: func.func @scaled_iota_reduce_other_dim
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<-100> : tensor<5xi64>
// CHECK-NEXT: return %[[C]]

// Non-constant init: still foldable to max(init, bound).
func.func @iota_max_dynamic_init(%init: tensor<i64>) -> tensor<i64> {
  %iota = stablehlo.iota dim = 0 : tensor<1000000000000xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<1000000000000xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @iota_max_dynamic_init
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<999999999999> : tensor<i64>
// CHECK-NEXT: %[[MAX:.+]] = stablehlo.maximum %arg0, %[[C]] : tensor<i64>
// CHECK-NEXT: return %[[MAX]]

// Float progression: max of 0.5 * i + 1.0 for i in [0, 8) is 4.5.
func.func @float_iota_max() -> tensor<f32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %half = stablehlo.constant dense<5.000000e-01> : tensor<8xf32>
  %one = stablehlo.constant dense<1.000000e+00> : tensor<8xf32>
  %iota = stablehlo.iota dim = 0 : tensor<8xf32>
  %0 = stablehlo.multiply %iota, %half : tensor<8xf32>
  %1 = stablehlo.add %0, %one : tensor<8xf32>
  %2 = stablehlo.reduce(%1 init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}
// CHECK-LABEL: func.func @float_iota_max
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<4.500000e+00> : tensor<f32>
// CHECK-NEXT: return %[[C]]

// Add reductions are not handled by this pattern.
func.func @iota_add_untouched() -> tensor<i64> {
  %init = stablehlo.constant dense<0> : tensor<i64>
  %iota = stablehlo.iota dim = 0 : tensor<10xi64>
  %0 = stablehlo.reduce(%iota init: %init) applies stablehlo.add across dimensions = [0] : (tensor<10xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @iota_add_untouched
// CHECK: stablehlo.reduce

func.func @main() {
  %r0 = func.call @issue_3530() : () -> tensor<i64>
  %e0 = stablehlo.constant dense<1200> : tensor<i64>
  check.expect_eq %r0, %e0 : tensor<i64>

  %r1 = func.call @iota_max_init_dominates() : () -> tensor<i64>
  %e1 = stablehlo.constant dense<100> : tensor<i64>
  check.expect_eq %r1, %e1 : tensor<i64>

  %r2 = func.call @scaled_iota_min() : () -> tensor<i32>
  %e2 = stablehlo.constant dense<5> : tensor<i32>
  check.expect_eq %r2, %e2 : tensor<i32>

  %r3 = func.call @negated_iota_min() : () -> tensor<i32>
  %e3 = stablehlo.constant dense<-6> : tensor<i32>
  check.expect_eq %r3, %e3 : tensor<i32>

  %r4 = func.call @iota_2d_full_reduce() : () -> tensor<i64>
  %e4 = stablehlo.constant dense<8> : tensor<i64>
  check.expect_eq %r4, %e4 : tensor<i64>

  %r5 = func.call @iota_2d_reduce_iota_dim() : () -> tensor<4xi64>
  %e5 = stablehlo.constant dense<8> : tensor<4xi64>
  check.expect_eq %r5, %e5 : tensor<4xi64>

  %r6 = func.call @iota_2d_reduce_other_dim() : () -> tensor<9xi64>
  %e6 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8]> : tensor<9xi64>
  check.expect_eq %r6, %e6 : tensor<9xi64>

  %r7 = func.call @iota_2d_reduce_other_dim_init_clamps() : () -> tensor<9xi64>
  %e7 = stablehlo.constant dense<[3, 3, 3, 3, 4, 5, 6, 7, 8]> : tensor<9xi64>
  check.expect_eq %r7, %e7 : tensor<9xi64>

  %r8 = func.call @scaled_iota_reduce_other_dim() : () -> tensor<5xi64>
  %e8 = stablehlo.constant dense<-100> : tensor<5xi64>
  check.expect_eq %r8, %e8 : tensor<5xi64>

  %r9 = func.call @float_iota_max() : () -> tensor<f32>
  %e9 = stablehlo.constant dense<4.500000e+00> : tensor<f32>
  check.expect_eq %r9, %e9 : tensor<f32>

  %r10 = func.call @iota_add_untouched() : () -> tensor<i64>
  %e10 = stablehlo.constant dense<45> : tensor<i64>
  check.expect_eq %r10, %e10 : tensor<i64>
  return
}
