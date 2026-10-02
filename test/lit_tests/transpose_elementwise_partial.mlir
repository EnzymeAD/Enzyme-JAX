// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=transpose_elementwise_transpose(1)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --inline --canonicalize --symbol-dce | stablehlo-translate --interpret

// Cancel the inverse transposes on the predicate and false value, moving the
// output transpose onto the one remaining operand. Three transposes become one.
// CHECK-LABEL: func.func @select_partial(
// CHECK-SAME: %[[P:.*]]: tensor<2x3xi1>, %[[X:.*]]: tensor<3x2xi32>, %[[Y:.*]]: tensor<2x3xi32>
// CHECK-NEXT: %[[TX:.*]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.select %[[P]], %[[TX]], %[[Y]] : tensor<2x3xi1>, tensor<2x3xi32>
// CHECK-NEXT: return %[[R]] : tensor<2x3xi32>
func.func @select_partial(%p: tensor<2x3xi1>, %x: tensor<3x2xi32>, %y: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %tp = stablehlo.transpose %p, dims = [1, 0] : (tensor<2x3xi1>) -> tensor<3x2xi1>
  %ty = stablehlo.transpose %y, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.select %tp, %x, %ty : tensor<3x2xi1>, tensor<3x2xi32>
  %r = stablehlo.transpose %s, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// Scalar predicates have no axes to transpose.
// CHECK-LABEL: func.func @select_scalar(
// CHECK-SAME: %[[P:.*]]: tensor<i1>, %[[X:.*]]: tensor<3x2xi32>, %[[Y:.*]]: tensor<2x3xi32>
// CHECK-NEXT: %[[TX:.*]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.select %[[P]], %[[TX]], %[[Y]] : tensor<i1>, tensor<2x3xi32>
// CHECK-NEXT: return %[[R]] : tensor<2x3xi32>
func.func @select_scalar(%p: tensor<i1>, %x: tensor<3x2xi32>, %y: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %ty = stablehlo.transpose %y, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.select %p, %x, %ty : tensor<i1>, tensor<3x2xi32>
  %r = stablehlo.transpose %s, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// Do not duplicate arithmetic that is also consumed in its original layout.
// CHECK-LABEL: func.func @shared_arithmetic(
// CHECK-SAME: %[[X:.*]]: tensor<2x3xi32>, %[[Y:.*]]: tensor<3x2xi32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
// CHECK-NEXT: %[[S:.*]] = stablehlo.subtract %[[T]], %[[Y]] : tensor<3x2xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.transpose %[[S]], dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: return %[[R]], %[[S]] : tensor<2x3xi32>, tensor<3x2xi32>
func.func @shared_arithmetic(%x: tensor<2x3xi32>, %y: tensor<3x2xi32>) -> (tensor<2x3xi32>, tensor<3x2xi32>) {
  %tx = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.subtract %tx, %y : tensor<3x2xi32>
  %r = stablehlo.transpose %s, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r, %s : tensor<2x3xi32>, tensor<3x2xi32>
}

// The inner transpose remains live through the second result. Moving the
// outer transpose would create two new transposes while removing only one.
// CHECK-LABEL: func.func @unprofitable_select(
// CHECK-SAME: %[[P:.*]]: tensor<3x2xi1>, %[[X:.*]]: tensor<2x3xi32>, %[[Y:.*]]: tensor<3x2xi32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
// CHECK-NEXT: %[[S:.*]] = stablehlo.select %[[P]], %[[T]], %[[Y]] : tensor<3x2xi1>, tensor<3x2xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.transpose %[[S]], dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: return %[[R]], %[[T]] : tensor<2x3xi32>, tensor<3x2xi32>
func.func @unprofitable_select(%p: tensor<3x2xi1>, %x: tensor<2x3xi32>, %y: tensor<3x2xi32>) -> (tensor<2x3xi32>, tensor<3x2xi32>) {
  %tx = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.select %p, %tx, %y : tensor<3x2xi1>, tensor<3x2xi32>
  %r = stablehlo.transpose %s, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r, %tx : tensor<2x3xi32>, tensor<3x2xi32>
}

// A shared inverse transpose cannot be removed. Keep a neutral rewrite from
// moving the output transpose back and forth between shared expressions.
// CHECK-LABEL: func.func @shared_inverse_operand(
// CHECK-SAME: %[[X:.*]]: tensor<2x3xi32>, %[[Y:.*]]: tensor<3x2xi32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
// CHECK-NEXT: %[[S:.*]] = stablehlo.subtract %[[T]], %[[Y]] : tensor<3x2xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.transpose %[[S]], dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: return %[[R]], %[[T]] : tensor<2x3xi32>, tensor<3x2xi32>
func.func @shared_inverse_operand(%x: tensor<2x3xi32>, %y: tensor<3x2xi32>) -> (tensor<2x3xi32>, tensor<3x2xi32>) {
  %tx = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.subtract %tx, %y : tensor<3x2xi32>
  %r = stablehlo.transpose %s, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r, %tx : tensor<2x3xi32>, tensor<3x2xi32>
}

// An inverse permutation need not equal the forward permutation.
// CHECK-LABEL: func.func @three_axes(
// CHECK-SAME: %[[X:.*]]: tensor<2x3x4xi32>, %[[Y:.*]]: tensor<2x3x4xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.subtract %[[X]], %[[Y]] : tensor<2x3x4xi32>
// CHECK-NEXT: return %[[R]] : tensor<2x3x4xi32>
func.func @three_axes(%x: tensor<2x3x4xi32>, %y: tensor<2x3x4xi32>) -> tensor<2x3x4xi32> {
  %tx = stablehlo.transpose %x, dims = [2, 0, 1] : (tensor<2x3x4xi32>) -> tensor<4x2x3xi32>
  %ty = stablehlo.transpose %y, dims = [2, 0, 1] : (tensor<2x3x4xi32>) -> tensor<4x2x3xi32>
  %s = stablehlo.subtract %tx, %ty : tensor<4x2x3xi32>
  %r = stablehlo.transpose %s, dims = [1, 2, 0] : (tensor<4x2x3xi32>) -> tensor<2x3x4xi32>
  return %r : tensor<2x3x4xi32>
}

// Partial cancellation must transpose the unmatched operand using the forward
// permutation, even when that permutation differs from its inverse.
// CHECK-LABEL: func.func @three_axes_partial(
// CHECK-SAME: %[[X:.*]]: tensor<2x3x4xi32>, %[[Y:.*]]: tensor<4x2x3xi32>
// CHECK-NEXT: %[[TY:.*]] = stablehlo.transpose %[[Y]], dims = [1, 2, 0] : (tensor<4x2x3xi32>) -> tensor<2x3x4xi32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.subtract %[[X]], %[[TY]] : tensor<2x3x4xi32>
// CHECK-NEXT: return %[[R]] : tensor<2x3x4xi32>
func.func @three_axes_partial(%x: tensor<2x3x4xi32>, %y: tensor<4x2x3xi32>) -> tensor<2x3x4xi32> {
  %tx = stablehlo.transpose %x, dims = [2, 0, 1] : (tensor<2x3x4xi32>) -> tensor<4x2x3xi32>
  %s = stablehlo.subtract %tx, %y : tensor<4x2x3xi32>
  %r = stablehlo.transpose %s, dims = [1, 2, 0] : (tensor<4x2x3xi32>) -> tensor<2x3x4xi32>
  return %r : tensor<2x3x4xi32>
}

func.func @main() {
  %p = stablehlo.constant dense<[[true, false, true], [false, true, false]]> : tensor<2x3xi1>
  %x = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %y = stablehlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi32>
  %expected = stablehlo.constant dense<[[1, 20, 5], [40, 4, 60]]> : tensor<2x3xi32>
  %r = func.call @select_partial(%p, %x, %y) : (tensor<2x3xi1>, tensor<3x2xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  check.expect_eq %r, %expected : tensor<2x3xi32>
  %yes = stablehlo.constant dense<true> : tensor<i1>
  %transposed = stablehlo.constant dense<[[1, 3, 5], [2, 4, 6]]> : tensor<2x3xi32>
  %s = func.call @select_scalar(%yes, %x, %y) : (tensor<i1>, tensor<3x2xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  check.expect_eq %s, %transposed : tensor<2x3xi32>
  return
}
