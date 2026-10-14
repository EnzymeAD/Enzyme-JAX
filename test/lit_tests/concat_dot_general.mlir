// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_dot_general" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Pieces contracted with the same %b, concatenated along a free dimension of
// the pieces: the pieces concatenated, contracted once.
func.func @lhs_free(%p0: tensor<8x3x4xf64>, %p1: tensor<8x3x4xf64>, %p2: tensor<8x2x4xf64>, %b: tensor<5x4xf64>) -> tensor<8x8x5xf64> {
  %d0 = stablehlo.dot_general %p0, %b, contracting_dims = [2] x [1] : (tensor<8x3x4xf64>, tensor<5x4xf64>) -> tensor<8x3x5xf64>
  %d1 = stablehlo.dot_general %p1, %b, contracting_dims = [2] x [1] : (tensor<8x3x4xf64>, tensor<5x4xf64>) -> tensor<8x3x5xf64>
  %d2 = stablehlo.dot_general %p2, %b, contracting_dims = [2] x [1] : (tensor<8x2x4xf64>, tensor<5x4xf64>) -> tensor<8x2x5xf64>
  %c = stablehlo.concatenate %d0, %d1, %d2, dim = 1 : (tensor<8x3x5xf64>, tensor<8x3x5xf64>, tensor<8x2x5xf64>) -> tensor<8x8x5xf64>
  return %c : tensor<8x8x5xf64>
}

// CHECK-LABEL: func.func @lhs_free(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x2x4xf64>, %[[V3:[^ ,:)]+]]: tensor<5x4xf64>) -> tensor<8x8x5xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], %[[V2]], dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>, tensor<8x2x4xf64>) -> tensor<8x8x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dot_general %[[V4]], %[[V3]], contracting_dims = [2] x [1] : (tensor<8x8x4xf64>, tensor<5x4xf64>) -> tensor<8x8x5xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x8x5xf64>
// CHECK-NEXT: }

// The same %a against different rhs pieces, along their free dimension.
func.func @rhs_free(%a: tensor<5x4xf64>, %q0: tensor<4x3xf64>, %q1: tensor<4x6xf64>) -> tensor<5x9xf64> {
  %d0 = stablehlo.dot_general %a, %q0, contracting_dims = [1] x [0] : (tensor<5x4xf64>, tensor<4x3xf64>) -> tensor<5x3xf64>
  %d1 = stablehlo.dot_general %a, %q1, contracting_dims = [1] x [0] : (tensor<5x4xf64>, tensor<4x6xf64>) -> tensor<5x6xf64>
  %c = stablehlo.concatenate %d0, %d1, dim = 1 : (tensor<5x3xf64>, tensor<5x6xf64>) -> tensor<5x9xf64>
  return %c : tensor<5x9xf64>
}

// CHECK-LABEL: func.func @rhs_free(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x4xf64>, %[[V1:[^ ,:)]+]]: tensor<4x3xf64>, %[[V2:[^ ,:)]+]]: tensor<4x6xf64>) -> tensor<5x9xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V1]], %[[V2]], dim = 1 : (tensor<4x3xf64>, tensor<4x6xf64>) -> tensor<4x9xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V3]], contracting_dims = [1] x [0] : (tensor<5x4xf64>, tensor<4x9xf64>) -> tensor<5x9xf64>
// CHECK-NEXT: return %[[V4]] : tensor<5x9xf64>
// CHECK-NEXT: }

// Batched dots of different operands, along the batching dimension.
func.func @batch(%a0: tensor<2x5x4xf64>, %a1: tensor<3x5x4xf64>, %q0: tensor<2x4x6xf64>, %q1: tensor<3x4x6xf64>) -> tensor<5x5x6xf64> {
  %d0 = stablehlo.dot_general %a0, %q0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x5x4xf64>, tensor<2x4x6xf64>) -> tensor<2x5x6xf64>
  %d1 = stablehlo.dot_general %a1, %q1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x5x4xf64>, tensor<3x4x6xf64>) -> tensor<3x5x6xf64>
  %c = stablehlo.concatenate %d0, %d1, dim = 0 : (tensor<2x5x6xf64>, tensor<3x5x6xf64>) -> tensor<5x5x6xf64>
  return %c : tensor<5x5x6xf64>
}

// CHECK-LABEL: func.func @batch(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x5x4xf64>, %[[V1:[^ ,:)]+]]: tensor<3x5x4xf64>, %[[V2:[^ ,:)]+]]: tensor<2x4x6xf64>, %[[V3:[^ ,:)]+]]: tensor<3x4x6xf64>) -> tensor<5x5x6xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<2x5x4xf64>, tensor<3x5x4xf64>) -> tensor<5x5x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 0 : (tensor<2x4x6xf64>, tensor<3x4x6xf64>) -> tensor<5x4x6xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.dot_general %[[V4]], %[[V5]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x5x4xf64>, tensor<5x4x6xf64>) -> tensor<5x5x6xf64>
// CHECK-NEXT: return %[[V6]] : tensor<5x5x6xf64>
// CHECK-NEXT: }

// Different rhs, concatenated along a free dimension of the lhs: left alone.
func.func @rhs_differs(%p0: tensor<8x3x4xf64>, %p1: tensor<8x3x4xf64>, %b0: tensor<5x4xf64>, %b1: tensor<5x4xf64>) -> tensor<8x6x5xf64> {
  %d0 = stablehlo.dot_general %p0, %b0, contracting_dims = [2] x [1] : (tensor<8x3x4xf64>, tensor<5x4xf64>) -> tensor<8x3x5xf64>
  %d1 = stablehlo.dot_general %p1, %b1, contracting_dims = [2] x [1] : (tensor<8x3x4xf64>, tensor<5x4xf64>) -> tensor<8x3x5xf64>
  %c = stablehlo.concatenate %d0, %d1, dim = 1 : (tensor<8x3x5xf64>, tensor<8x3x5xf64>) -> tensor<8x6x5xf64>
  return %c : tensor<8x6x5xf64>
}

// CHECK-LABEL: func.func @rhs_differs(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V2:[^ ,:)]+]]: tensor<5x4xf64>, %[[V3:[^ ,:)]+]]: tensor<5x4xf64>) -> tensor<8x6x5xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V2]], contracting_dims = [2] x [1] : (tensor<8x3x4xf64>, tensor<5x4xf64>) -> tensor<8x3x5xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dot_general %[[V1]], %[[V3]], contracting_dims = [2] x [1] : (tensor<8x3x4xf64>, tensor<5x4xf64>) -> tensor<8x3x5xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 1 : (tensor<8x3x5xf64>, tensor<8x3x5xf64>) -> tensor<8x6x5xf64>
// CHECK-NEXT: return %[[V6]] : tensor<8x6x5xf64>
// CHECK-NEXT: }
