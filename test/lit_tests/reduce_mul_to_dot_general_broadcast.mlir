// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_mul_to_dot_general" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A kept dimension that only the rhs's broadcast expands is a free dimension
// of the lhs, not a batching dimension: the broadcast loses it and the result
// is transposed back into the reduce's order.
func.func @one_side(%x: tensor<4x8x3x6xf64>, %b: tensor<4x6xf64>) -> tensor<4x8x3xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0, 3] : (tensor<4x6xf64>) -> tensor<4x8x3x6xf64>
  %m = stablehlo.multiply %x, %bb : tensor<4x8x3x6xf64>
  %r = stablehlo.reduce(%m init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<4x8x3x6xf64>, tensor<f64>) -> tensor<4x8x3xf64>
  return %r : tensor<4x8x3xf64>
}

// Neither operand broadcasts a kept dimension: every kept dimension batches,
// as before.
func.func @batched(%x: tensor<4x8x6xf64>, %y: tensor<4x8x6xf64>) -> tensor<4x8xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %m = stablehlo.multiply %x, %y : tensor<4x8x6xf64>
  %r = stablehlo.reduce(%m init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<4x8x6xf64>, tensor<f64>) -> tensor<4x8xf64>
  return %r : tensor<4x8xf64>
}

// CHECK-LABEL: func.func @one_side(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<4x8x3x6xf64>, %[[B:[^ ,:)]+]]: tensor<4x6xf64>) -> tensor<4x8x3xf64> {
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[X]], %[[B]], batching_dims = [0] x [0], contracting_dims = [3] x [1] : (tensor<4x8x3x6xf64>, tensor<4x6xf64>) -> tensor<4x8x3xf64>
// CHECK-NEXT: return %[[D]] : tensor<4x8x3xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @batched(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<4x8x6xf64>, %[[Y:[^ ,:)]+]]: tensor<4x8x6xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[X]], %[[Y]], batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2] : (tensor<4x8x6xf64>, tensor<4x8x6xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: return %[[D]] : tensor<4x8xf64>
// CHECK-NEXT: }
