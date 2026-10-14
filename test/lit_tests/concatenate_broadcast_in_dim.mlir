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

// Each operand repeated 3 times along the concatenated dimension: stacked,
// broadcast, and the stack and repeat dimensions merged.
func.func @broadcast_dim(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4x6xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %c = stablehlo.concatenate %ba, %bb, dim = 1 : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x6xf64>
  return %c : tensor<4x6xf64>
}

// CHECK-LABEL: func.func @broadcast_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>) -> tensor<4x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 0 : (tensor<1x4xf64>, tensor<1x4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [1, 0] : (tensor<2x4xf64>) -> tensor<4x2x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<4x2x3xf64>) -> tensor<4x6xf64>
// CHECK-NEXT: return %[[V6]] : tensor<4x6xf64>
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

// Three pieces, each repeating its operand 3 times along dimension 2: element
// k * 3 + j of dimension 2 is %x_k.
func.func @pieces(%a: tensor<8x4x4xf64>, %b: tensor<8x4x4xf64>, %c: tensor<8x4x4xf64>) -> tensor<8x4x9x4xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [0, 3, 1] : (tensor<8x4x4xf64>) -> tensor<8x4x3x4xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0, 3, 1] : (tensor<8x4x4xf64>) -> tensor<8x4x3x4xf64>
  %bc = stablehlo.broadcast_in_dim %c, dims = [0, 3, 1] : (tensor<8x4x4xf64>) -> tensor<8x4x3x4xf64>
  %r = stablehlo.concatenate %ba, %bb, %bc, dim = 2 : (tensor<8x4x3x4xf64>, tensor<8x4x3x4xf64>, tensor<8x4x3x4xf64>) -> tensor<8x4x9x4xf64>
  return %r : tensor<8x4x9x4xf64>
}

// CHECK-LABEL: func.func @pieces(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x4x4xf64>) -> tensor<8x4x9x4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], %[[V5]], dim = 0 : (tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V6]], dims = [2, 0, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<8x4x3x3x4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<8x4x3x3x4xf64>) -> tensor<8x4x9x4xf64>
// CHECK-NEXT: return %[[V8]] : tensor<8x4x9x4xf64>
// CHECK-NEXT: }

// The pieces repeat their operands a different number of times: left alone.
func.func @uneven(%a: tensor<5xf64>, %b: tensor<5xf64>) -> tensor<5x5xf64> {
  %ba = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<5xf64>) -> tensor<5x2xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<5xf64>) -> tensor<5x3xf64>
  %r = stablehlo.concatenate %ba, %bb, dim = 1 : (tensor<5x2xf64>, tensor<5x3xf64>) -> tensor<5x5xf64>
  return %r : tensor<5x5xf64>
}

// CHECK-LABEL: func.func @uneven(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5xf64>, %[[V1:[^ ,:)]+]]: tensor<5xf64>) -> tensor<5x5xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0] : (tensor<5xf64>) -> tensor<5x2xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0] : (tensor<5xf64>) -> tensor<5x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 1 : (tensor<5x2xf64>, tensor<5x3xf64>) -> tensor<5x5xf64>
// CHECK-NEXT: return %[[V4]] : tensor<5x5xf64>
// CHECK-NEXT: }

// The concatenated dimension comes from a unit operand dimension the
// broadcasts expand (scalars of a basis spread over 5 rows each): the same as
// an inserted one.
func.func @expanded_unit(%s0: tensor<1x1xf64>, %s1: tensor<1x1xf64>, %s2: tensor<1x1xf64>) -> tensor<8x15x5xf64> {
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x5x5xf64>
  %b1 = stablehlo.broadcast_in_dim %s1, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x5x5xf64>
  %b2 = stablehlo.broadcast_in_dim %s2, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x5x5xf64>
  %c = stablehlo.concatenate %b0, %b1, %b2, dim = 1 : (tensor<8x5x5xf64>, tensor<8x5x5xf64>, tensor<8x5x5xf64>) -> tensor<8x15x5xf64>
  return %c : tensor<8x15x5xf64>
}

// CHECK-LABEL: func.func @expanded_unit(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1x1xf64>, %[[V1:[^ ,:)]+]]: tensor<1x1xf64>, %[[V2:[^ ,:)]+]]: tensor<1x1xf64>) -> tensor<8x15x5xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<1x1xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<1x1xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<1x1xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], %[[V5]], dim = 0 : (tensor<1x1x1xf64>, tensor<1x1x1xf64>, tensor<1x1x1xf64>) -> tensor<3x1x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V6]], dims = [1, 3, 2] : (tensor<3x1x1xf64>) -> tensor<8x3x5x5xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<8x3x5x5xf64>) -> tensor<8x15x5xf64>
// CHECK-NEXT: return %[[V8]] : tensor<8x15x5xf64>
// CHECK-NEXT: }
