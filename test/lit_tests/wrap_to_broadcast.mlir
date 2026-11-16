// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=wrap_to_broadcast" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Wrapping 3 rows of a 3-row tensor on both sides: three copies of it.
func.func @whole(%x: tensor<3x4x1xf64>) -> tensor<9x4x1xf64> {
  %r = "enzymexla.wrap"(%x) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 3 : i64}> : (tensor<3x4x1xf64>) -> tensor<9x4x1xf64>
  return %r : tensor<9x4x1xf64>
}

// CHECK-LABEL: func.func @whole(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4x1xf64>) -> tensor<9x4x1xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<3x4x1xf64>) -> tensor<1x3x4x1xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1, 2, 3] : (tensor<1x3x4x1xf64>) -> tensor<3x3x4x1xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<3x3x4x1xf64>) -> tensor<9x4x1xf64>
// CHECK-NEXT: return %[[V3]] : tensor<9x4x1xf64>
// CHECK-NEXT: }

// A wrapped dimension of size 1: a broadcast along it.
func.func @size_one(%x: tensor<5x1xf64>) -> tensor<5x3xf64> {
  %r = "enzymexla.wrap"(%x) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<5x1xf64>) -> tensor<5x3xf64>
  return %r : tensor<5x3xf64>
}

// CHECK-LABEL: func.func @size_one(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x1xf64>) -> tensor<5x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<5x1xf64>) -> tensor<5x1x1xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1, 2] : (tensor<5x1x1xf64>) -> tensor<5x3x1xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<5x3x1xf64>) -> tensor<5x3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<5x3xf64>
// CHECK-NEXT: }

// Only part of the tensor on one side: left alone.
func.func @partial(%x: tensor<3x4xf64>) -> tensor<4x4xf64> {
  %r = "enzymexla.wrap"(%x) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<3x4xf64>) -> tensor<4x4xf64>
  return %r : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @partial(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>) -> tensor<4x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = "enzymexla.wrap"(%[[V0]]) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<3x4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT: return %[[V1]] : tensor<4x4xf64>
// CHECK-NEXT: }
