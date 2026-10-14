// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Three sums over dimension 2, concatenated along dimension 1 (input
// dimension 1): one sum of the concatenated inputs.
func.func @kept_dim(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>, %c: tensor<8x2x4xf64>) -> tensor<8x8xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rc = stablehlo.reduce(%c init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x2x4xf64>, tensor<f64>) -> tensor<8x2xf64>
  %r = stablehlo.concatenate %ra, %rb, %rc, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>, tensor<8x2xf64>) -> tensor<8x8xf64>
  return %r : tensor<8x8xf64>
}

// CHECK-LABEL: func.func @kept_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x2x4xf64>) -> tensor<8x8xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], %[[V2]], dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>, tensor<8x2x4xf64>) -> tensor<8x8x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reduce(%[[V4]] init: %[[V3]]) applies stablehlo.add across dimensions = [2] : (tensor<8x8x4xf64>, tensor<f64>) -> tensor<8x8xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x8xf64>
// CHECK-NEXT: }

// The reduced dimension comes first: result dimension 1 is input dimension 2.
func.func @after_reduced(%a: tensor<4x8x3xf64>, %b: tensor<4x8x3xf64>) -> tensor<8x6xf64> {
  %z = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8x3xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8x3xf64>, tensor<f64>) -> tensor<8x3xf64>
  %r = stablehlo.concatenate %ra, %rb, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
  return %r : tensor<8x6xf64>
}

// CHECK-LABEL: func.func @after_reduced(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x3xf64>, %[[V1:[^ ,:)]+]]: tensor<4x8x3xf64>) -> tensor<8x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 2 : (tensor<4x8x3xf64>, tensor<4x8x3xf64>) -> tensor<4x8x6xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V3]] init: %[[V2]]) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8x6xf64>, tensor<f64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x6xf64>
// CHECK-NEXT: }

// Different reductions (add and max): left alone.
func.func @different_body(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> tensor<8x6xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.maximum across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %r = stablehlo.concatenate %ra, %rb, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
  return %r : tensor<8x6xf64>
}

// CHECK-LABEL: func.func @different_body(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>) -> tensor<8x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V1]] init: %[[V2]]) applies stablehlo.maximum across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x6xf64>
// CHECK-NEXT: }

// A reduce also used on its own: left alone.
func.func @other_user(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> (tensor<8x6xf64>, tensor<8x3xf64>) {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %r = stablehlo.concatenate %ra, %rb, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
  return %r, %ra : tensor<8x6xf64>, tensor<8x3xf64>
}

// CHECK-LABEL: func.func @other_user(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>) -> (tensor<8x6xf64>, tensor<8x3xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V1]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %[[V5]], %[[V3]] : tensor<8x6xf64>, tensor<8x3xf64>
// CHECK-NEXT: }
