// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=select_to_logical" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s

// select(p, x, false) -> and(p, x)
func.func @select_x_false(%p: tensor<4xi1>, %x: tensor<4xi1>) -> tensor<4xi1> {
  %false = stablehlo.constant dense<false> : tensor<4xi1>
  %r = stablehlo.select %p, %x, %false : tensor<4xi1>, tensor<4xi1>
  return %r : tensor<4xi1>
}
// CHECK-LABEL: func.func @select_x_false
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.and %arg0, %arg1 : tensor<4xi1>

// -----

// select(p, true, x) -> or(p, x)
func.func @select_true_x(%p: tensor<4xi1>, %x: tensor<4xi1>) -> tensor<4xi1> {
  %true = stablehlo.constant dense<true> : tensor<4xi1>
  %r = stablehlo.select %p, %true, %x : tensor<4xi1>, tensor<4xi1>
  return %r : tensor<4xi1>
}
// CHECK-LABEL: func.func @select_true_x
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.or %arg0, %arg1 : tensor<4xi1>

// -----

// select(p, false, x) -> and(not(p), x)
func.func @select_false_x(%p: tensor<4xi1>, %x: tensor<4xi1>) -> tensor<4xi1> {
  %false = stablehlo.constant dense<false> : tensor<4xi1>
  %r = stablehlo.select %p, %false, %x : tensor<4xi1>, tensor<4xi1>
  return %r : tensor<4xi1>
}
// CHECK-LABEL: func.func @select_false_x
// CHECK-NOT: stablehlo.select
// CHECK: %[[N:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK: stablehlo.and %[[N]], %arg1 : tensor<4xi1>

// -----

// select(p, x, true) -> or(not(p), x)
func.func @select_x_true(%p: tensor<4xi1>, %x: tensor<4xi1>) -> tensor<4xi1> {
  %true = stablehlo.constant dense<true> : tensor<4xi1>
  %r = stablehlo.select %p, %x, %true : tensor<4xi1>, tensor<4xi1>
  return %r : tensor<4xi1>
}
// CHECK-LABEL: func.func @select_x_true
// CHECK-NOT: stablehlo.select
// CHECK: %[[N:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK: stablehlo.or %[[N]], %arg1 : tensor<4xi1>

// -----

// A rank-0 predicate against a ranked result relies on select's implicit
// broadcast, which and/or don't have -- must not fire.
func.func @scalar_pred(%p: tensor<i1>, %x: tensor<4xi1>) -> tensor<4xi1> {
  %false = stablehlo.constant dense<false> : tensor<4xi1>
  %r = stablehlo.select %p, %x, %false : tensor<i1>, tensor<4xi1>
  return %r : tensor<4xi1>
}
// CHECK-LABEL: func.func @scalar_pred
// CHECK: stablehlo.select

// -----

// Non-i1 operands are not boolean logic -- must not fire.
func.func @not_i1(%p: tensor<4xi1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %r = stablehlo.select %p, %x, %zero : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @not_i1
// CHECK: stablehlo.select

// -----

// A non-splat i1 constant branch is not an all-true/all-false -- must not fire.
func.func @non_splat(%p: tensor<4xi1>, %x: tensor<4xi1>) -> tensor<4xi1> {
  %c = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %r = stablehlo.select %p, %x, %c : tensor<4xi1>, tensor<4xi1>
  return %r : tensor<4xi1>
}
// CHECK-LABEL: func.func @non_splat
// CHECK: stablehlo.select
