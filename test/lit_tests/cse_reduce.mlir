// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_reduce;concat_to_broadcast" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Identical reductions are one value, so a concatenate of them repeats it.
func.func @repeat(%x: tensor<6x8x5xf64>) -> tensor<8x15xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %r0 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
  %r1 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
  %r2 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
  %c = stablehlo.concatenate %r0, %r1, %r2, dim = 1 : (tensor<8x5xf64>, tensor<8x5xf64>, tensor<8x5xf64>) -> tensor<8x15xf64>
  return %c : tensor<8x15xf64>
}

// CHECK-LABEL: func.func @repeat(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<6x8x5xf64>) -> tensor<8x15xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V1]]) applies stablehlo.add across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x5xf64>) -> tensor<8x1x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [0, 1, 2] : (tensor<8x1x5xf64>) -> tensor<8x3x5xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<8x3x5xf64>) -> tensor<8x15xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x15xf64>
// CHECK-NEXT: }

// Different reductions (another body, other dimensions) stay apart.
func.func @different(%x: tensor<6x8x5xf64>) -> (tensor<8x5xf64>, tensor<8x5xf64>, tensor<6x5xf64>) {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %r0 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
  %r1 = stablehlo.reduce(%x init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
  %r2 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<6x5xf64>
  return %r0, %r1, %r2 : tensor<8x5xf64>, tensor<8x5xf64>, tensor<6x5xf64>
}

// CHECK-LABEL: func.func @different(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<6x8x5xf64>) -> (tensor<8x5xf64>, tensor<8x5xf64>, tensor<6x5xf64>) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V1]]) applies stablehlo.add across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V1]]) applies stablehlo.maximum across dimensions = [0] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<8x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V1]]) applies stablehlo.add across dimensions = [1] : (tensor<6x8x5xf64>, tensor<f64>) -> tensor<6x5xf64>
// CHECK-NEXT: return %[[V2]], %[[V3]], %[[V4]] : tensor<8x5xf64>, tensor<8x5xf64>, tensor<6x5xf64>
// CHECK-NEXT: }
