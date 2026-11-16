// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_unbatch_broadcast" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A contraction formed from reduce(mul(broadcast(b), broadcast(y))): every
// dimension is batched, though b is only broadcast along y's and y along b's.
// Unbatched: a plain contraction of y with b.
func.func @both_sides(%b: tensor<3x4xf64>, %y: tensor<10x27x4xf64>) -> tensor<10x27x3xf64> {
  %bb = stablehlo.broadcast_in_dim %b, dims = [0, 1] : (tensor<3x4xf64>) -> tensor<3x4x10x27xf64>
  %yb = stablehlo.broadcast_in_dim %y, dims = [0, 1, 2] : (tensor<10x27x4xf64>) -> tensor<10x27x4x3xf64>
  %d = stablehlo.dot_general %bb, %yb, batching_dims = [2, 3, 0] x [0, 1, 3], contracting_dims = [1] x [2] : (tensor<3x4x10x27xf64>, tensor<10x27x4x3xf64>) -> tensor<10x27x3xf64>
  return %d : tensor<10x27x3xf64>
}

// CHECK-LABEL: func.func @both_sides(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<10x27x4xf64>) -> tensor<10x27x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V1]], contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<10x27x4xf64>) -> tensor<3x10x27xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [2, 0, 1] : (tensor<3x10x27xf64>) -> tensor<27x3x10xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [2, 0, 1] : (tensor<27x3x10xf64>) -> tensor<10x27x3xf64>
// CHECK-NEXT: return %[[V4]] : tensor<10x27x3xf64>
// CHECK-NEXT: }

// Only the lhs is a broadcast, along the middle batching dimension.
func.func @lhs_only(%b: tensor<5x4xf64>, %y: tensor<5x6x4x2xf64>) -> tensor<5x6x2xf64> {
  %bb = stablehlo.broadcast_in_dim %b, dims = [0, 2] : (tensor<5x4xf64>) -> tensor<5x6x4xf64>
  %d = stablehlo.dot_general %bb, %y, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2] : (tensor<5x6x4xf64>, tensor<5x6x4x2xf64>) -> tensor<5x6x2xf64>
  return %d : tensor<5x6x2xf64>
}

// CHECK-LABEL: func.func @lhs_only(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x4xf64>, %[[V1:[^ ,:)]+]]: tensor<5x6x4x2xf64>) -> tensor<5x6x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V1]], batching_dims = [0] x [0], contracting_dims = [1] x [2] : (tensor<5x4xf64>, tensor<5x6x4x2xf64>) -> tensor<5x6x2xf64>
// CHECK-NEXT: return %[[V2]] : tensor<5x6x2xf64>
// CHECK-NEXT: }

// Both operands are broadcasts along the batching dimension: left to
// dot_general_broadcast_in_dim.
func.func @both_broadcast(%a: tensor<4xf64>, %c: tensor<4xf64>) -> tensor<6xf64> {
  %ab = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<4xf64>) -> tensor<6x4xf64>
  %cb = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<4xf64>) -> tensor<6x4xf64>
  %d = stablehlo.dot_general %ab, %cb, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<6x4xf64>, tensor<6x4xf64>) -> tensor<6xf64>
  return %d : tensor<6xf64>
}

// CHECK-LABEL: func.func @both_broadcast(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>) -> tensor<6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [1] : (tensor<4xf64>) -> tensor<6x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [1] : (tensor<4xf64>) -> tensor<6x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dot_general %[[V2]], %[[V3]], batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<6x4xf64>, tensor<6x4xf64>) -> tensor<6xf64>
// CHECK-NEXT: return %[[V4]] : tensor<6xf64>
// CHECK-NEXT: }
