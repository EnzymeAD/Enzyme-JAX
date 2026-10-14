// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concatenate_broadcast_in_dim" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Rank 1: concatenated along the operand's only dimension.
func.func @vector(%a: tensor<3xf64>, %b: tensor<5xf64>) -> tensor<4x8xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xf64>) -> tensor<4x3xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<5xf64>) -> tensor<4x5xf64>
  %c = stablehlo.concatenate %ba, %bb, dim = 1 : (tensor<4x3xf64>, tensor<4x5xf64>) -> tensor<4x8xf64>
  return %c : tensor<4x8xf64>
}

// CHECK-LABEL: func.func @vector(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3xf64>, %[[V1:[^ ,:)]+]]: tensor<5xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<3xf64>, tensor<5xf64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x8xf64>
// CHECK-NEXT: }

// Rank 3, permuted: result dimension 2 is operand dimension 1.
func.func @permuted(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>, %c: tensor<8x2x4xf64>) -> tensor<8x4x8x3xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [0, 2, 1] : (tensor<8x3x4xf64>) -> tensor<8x4x3x3xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0, 2, 1] : (tensor<8x3x4xf64>) -> tensor<8x4x3x3xf64>
  %bc = stablehlo.broadcast_in_dim %c, dims = [0, 2, 1] : (tensor<8x2x4xf64>) -> tensor<8x4x2x3xf64>
  %r = stablehlo.concatenate %ba, %bb, %bc, dim = 2 : (tensor<8x4x3x3xf64>, tensor<8x4x3x3xf64>, tensor<8x4x2x3xf64>) -> tensor<8x4x8x3xf64>
  return %r : tensor<8x4x8x3xf64>
}

// CHECK-LABEL: func.func @permuted(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x2x4xf64>) -> tensor<8x4x8x3xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], %[[V2]], dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>, tensor<8x2x4xf64>) -> tensor<8x8x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [0, 2, 1] : (tensor<8x8x4xf64>) -> tensor<8x4x8x3xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x4x8x3xf64>
// CHECK-NEXT: }

// The concatenated dimension is a broadcast one: left alone.
func.func @broadcast_dim(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4x6xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %c = stablehlo.concatenate %ba, %bb, dim = 1 : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x6xf64>
  return %c : tensor<4x6xf64>
}

// CHECK-LABEL: func.func @broadcast_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>) -> tensor<4x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 1 : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x6xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4x6xf64>
// CHECK-NEXT: }

// One operand expands a dimension the other has in full: left alone.
func.func @expanded_other_dim(%a: tensor<1x3xf64>, %b: tensor<4x3xf64>) -> tensor<4x6xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x3xf64>) -> tensor<4x3xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0, 1] : (tensor<4x3xf64>) -> tensor<4x3xf64>
  %c = stablehlo.concatenate %ba, %bb, dim = 1 : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x6xf64>
  return %c : tensor<4x6xf64>
}

// CHECK-LABEL: func.func @expanded_other_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1x3xf64>, %[[V1:[^ ,:)]+]]: tensor<4x3xf64>) -> tensor<4x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 1] : (tensor<1x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1] : (tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 1 : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x6xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4x6xf64>
// CHECK-NEXT: }
