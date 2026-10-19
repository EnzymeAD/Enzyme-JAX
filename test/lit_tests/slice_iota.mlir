// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_iota" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The live lanes of a padded lane iota: an iota of their shape.
func.func @prefix() -> tensor<3x3x1xi64> {
  %i = stablehlo.iota dim = 1 : tensor<24x24x1xi64>
  %s = stablehlo.slice %i [0:3, 0:3, 0:1] : (tensor<24x24x1xi64>) -> tensor<3x3x1xi64>
  return %s : tensor<3x3x1xi64>
}

// Starting past zero along the iota's dimension, with a stride.
func.func @offset_stride() -> tensor<4x2xi32> {
  %i = stablehlo.iota dim = 0 : tensor<16x8xi32>
  %s = stablehlo.slice %i [3:11:2, 1:3] : (tensor<16x8xi32>) -> tensor<4x2xi32>
  return %s : tensor<4x2xi32>
}

// A float iota is left alone.
func.func @float() -> tensor<3xf32> {
  %i = stablehlo.iota dim = 0 : tensor<8xf32>
  %s = stablehlo.slice %i [2:5] : (tensor<8xf32>) -> tensor<3xf32>
  return %s : tensor<3xf32>
}

// CHECK-LABEL: func.func @prefix(
// CHECK-SAME: ) -> tensor<3x3x1xi64> {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.iota dim = 1 : tensor<3x3x1xi64>
// CHECK-NEXT: return %[[V0]] : tensor<3x3x1xi64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @offset_stride(
// CHECK-SAME: ) -> tensor<4x2xi32> {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<4x2xi32>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<4x2xi32>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4x2xi32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[V1]] : tensor<4x2xi32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.add %[[V3]], %[[V0]] : tensor<4x2xi32>
// CHECK-NEXT: return %[[V4]] : tensor<4x2xi32>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @float(
// CHECK-SAME: ) -> tensor<3xf32> {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<8xf32>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [2:5] : (tensor<8xf32>) -> tensor<3xf32>
// CHECK-NEXT: return %[[V1]] : tensor<3xf32>
// CHECK-NEXT: }
