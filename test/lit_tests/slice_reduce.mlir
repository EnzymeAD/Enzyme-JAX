// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Rows 0:3 of a reduce over dimension 1: the reduce of rows 0:3.
func.func @kept_rows(%x: tensor<8x4x5xf64>) -> tensor<8x3xf64> {
  %c = stablehlo.constant dense<0.0> : tensor<f64>
  %r = stablehlo.reduce(%x init: %c) applies stablehlo.add across dimensions = [2] : (tensor<8x4x5xf64>, tensor<f64>) -> tensor<8x4xf64>
  %s = stablehlo.slice %r [0:8, 0:3] : (tensor<8x4xf64>) -> tensor<8x3xf64>
  return %s : tensor<8x3xf64>
}

// CHECK-LABEL: func.func @kept_rows(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x4x5xf64>) -> tensor<8x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 0:3, 0:5] : (tensor<8x4x5xf64>) -> tensor<8x3x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V2]] init: %[[V1]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x5xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x3xf64>
// CHECK-NEXT: }

// The kept dimensions around a reduced one, with a stride.
func.func @strided(%x: tensor<6x4x8xf64>) -> tensor<3x4xf64> {
  %c = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
  %r = stablehlo.reduce(%x init: %c) applies stablehlo.maximum across dimensions = [1] : (tensor<6x4x8xf64>, tensor<f64>) -> tensor<6x8xf64>
  %s = stablehlo.slice %r [0:6:2, 1:8:2] : (tensor<6x8xf64>) -> tensor<3x4xf64>
  return %s : tensor<3x4xf64>
}

// CHECK-LABEL: func.func @strided(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<6x4x8xf64>) -> tensor<3x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:6:2, 0:4, 1:8:2] : (tensor<6x4x8xf64>) -> tensor<3x4x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V2]] init: %[[V1]]) applies stablehlo.maximum across dimensions = [1] : (tensor<3x4x4xf64>, tensor<f64>) -> tensor<3x4xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3x4xf64>
// CHECK-NEXT: }

// The reduce is also used whole: left alone.
func.func @other_user(%x: tensor<8x4x5xf64>) -> (tensor<8x3xf64>, tensor<8x4xf64>) {
  %c = stablehlo.constant dense<0.0> : tensor<f64>
  %r = stablehlo.reduce(%x init: %c) applies stablehlo.add across dimensions = [2] : (tensor<8x4x5xf64>, tensor<f64>) -> tensor<8x4xf64>
  %s = stablehlo.slice %r [0:8, 0:3] : (tensor<8x4xf64>) -> tensor<8x3xf64>
  return %s, %r : tensor<8x3xf64>, tensor<8x4xf64>
}

// CHECK-LABEL: func.func @other_user(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x4x5xf64>) -> (tensor<8x3xf64>, tensor<8x4xf64>) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V1]]) applies stablehlo.add across dimensions = [2] : (tensor<8x4x5xf64>, tensor<f64>) -> tensor<8x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:8, 0:3] : (tensor<8x4xf64>) -> tensor<8x3xf64>
// CHECK-NEXT: return %[[V3]], %[[V2]] : tensor<8x3xf64>, tensor<8x4xf64>
// CHECK-NEXT: }
