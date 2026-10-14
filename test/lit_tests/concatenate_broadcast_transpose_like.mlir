// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concatenate_broadcast_in_dim;transpose_like_broadcast_concat" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Broadcasts that only permute and insert a unit dimension:
// transpose_like_broadcast_concat moves such a broadcast onto the operands of
// a concatenate, and concatenate_broadcast_in_dim leaves them there instead of
// pulling it back out (the two used to undo each other forever).
func.func @transpose_like(%a: tensor<4x3xf64>, %b: tensor<4x5xf64>) -> tensor<8x1x4xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [2, 0] : (tensor<4x3xf64>) -> tensor<3x1x4xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [2, 0] : (tensor<4x5xf64>) -> tensor<5x1x4xf64>
  %c = stablehlo.concatenate %ba, %bb, dim = 0 : (tensor<3x1x4xf64>, tensor<5x1x4xf64>) -> tensor<8x1x4xf64>
  return %c : tensor<8x1x4xf64>
}

// CHECK-LABEL: func.func @transpose_like(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x3xf64>, %[[V1:[^ ,:)]+]]: tensor<4x5xf64>) -> tensor<8x1x4xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [2, 0] : (tensor<4x3xf64>) -> tensor<3x1x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [2, 0] : (tensor<4x5xf64>) -> tensor<5x1x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 0 : (tensor<3x1x4xf64>, tensor<5x1x4xf64>) -> tensor<8x1x4xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x1x4xf64>
// CHECK-NEXT: }

// A broadcast that expands a dimension is still pulled out of the concatenate.
func.func @expanding(%a: tensor<3xf64>, %b: tensor<5xf64>) -> tensor<4x8xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xf64>) -> tensor<4x3xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<5xf64>) -> tensor<4x5xf64>
  %c = stablehlo.concatenate %ba, %bb, dim = 1 : (tensor<4x3xf64>, tensor<4x5xf64>) -> tensor<4x8xf64>
  return %c : tensor<4x8xf64>
}

// CHECK-LABEL: func.func @expanding(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3xf64>, %[[V1:[^ ,:)]+]]: tensor<5xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<3xf64>, tensor<5xf64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x8xf64>
// CHECK-NEXT: }
