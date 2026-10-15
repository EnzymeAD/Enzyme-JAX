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

// Each dot's result is transposed the same way before the concatenate: one
// dot of the concatenated pieces, transposed once.
func.func @through_transpose(%a: tensor<3x4xf64>, %x0: tensor<8x4x4xf64>, %x1: tensor<8x4x4xf64>) -> tensor<16x3x4xf64> {
  %d0 = stablehlo.dot_general %a, %x0, contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<8x4x4xf64>) -> tensor<3x8x4xf64>
  %d1 = stablehlo.dot_general %a, %x1, contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<8x4x4xf64>) -> tensor<3x8x4xf64>
  %t0 = stablehlo.transpose %d0, dims = [1, 0, 2] : (tensor<3x8x4xf64>) -> tensor<8x3x4xf64>
  %t1 = stablehlo.transpose %d1, dims = [1, 0, 2] : (tensor<3x8x4xf64>) -> tensor<8x3x4xf64>
  %c = stablehlo.concatenate %t0, %t1, dim = 0 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>) -> tensor<16x3x4xf64>
  return %c : tensor<16x3x4xf64>
}

// CHECK-LABEL: func.func @through_transpose(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x4x4xf64>) -> tensor<16x3x4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V1]], %[[V2]], dim = 0 : (tensor<8x4x4xf64>, tensor<8x4x4xf64>) -> tensor<16x4x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V3]], contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<16x4x4xf64>) -> tensor<3x16x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V4]], dims = [1, 0, 2] : (tensor<3x16x4xf64>) -> tensor<16x3x4xf64>
// CHECK-NEXT: return %[[V5]] : tensor<16x3x4xf64>
// CHECK-NEXT: }

// The same %b against three pieces, concatenated along %b's rows: the pieces
// stacked, one dot, and the stack merged into the rows (stack-major).
func.func @stack_shared(%b: tensor<3x4xf64>, %x0: tensor<8x4xf64>, %x1: tensor<8x4xf64>, %x2: tensor<8x4xf64>) -> tensor<9x8xf64> {
  %d0 = stablehlo.dot_general %b, %x0, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4xf64>) -> tensor<3x8xf64>
  %d1 = stablehlo.dot_general %b, %x1, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4xf64>) -> tensor<3x8xf64>
  %d2 = stablehlo.dot_general %b, %x2, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4xf64>) -> tensor<3x8xf64>
  %c = stablehlo.concatenate %d0, %d1, %d2, dim = 0 : (tensor<3x8xf64>, tensor<3x8xf64>, tensor<3x8xf64>) -> tensor<9x8xf64>
  return %c : tensor<9x8xf64>
}

// CHECK-LABEL: func.func @stack_shared(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x4xf64>, %[[V3:[^ ,:)]+]]: tensor<8x4xf64>) -> tensor<9x8xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.concatenate %[[V4]], %[[V5]], %[[V6]], dim = 0 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<3x8x4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V7]], contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<3x8x4xf64>) -> tensor<3x3x8xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.transpose %[[V8]], dims = [1, 0, 2] : (tensor<3x3x8xf64>) -> tensor<3x3x8xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.reshape %[[V9]] : (tensor<3x3x8xf64>) -> tensor<9x8xf64>
// CHECK-NEXT: return %[[V10]] : tensor<9x8xf64>
// CHECK-NEXT: }

// The same through a transpose, as BK2PARTIAL's output: pieces transposed to
// [e, i, k], concatenated along i.
func.func @stack_through_transpose(%b: tensor<3x4xf64>, %x0: tensor<8x4x4xf64>, %x1: tensor<8x4x4xf64>) -> tensor<8x6x4xf64> {
  %d0 = stablehlo.dot_general %b, %x0, contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<8x4x4xf64>) -> tensor<3x8x4xf64>
  %d1 = stablehlo.dot_general %b, %x1, contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<8x4x4xf64>) -> tensor<3x8x4xf64>
  %t0 = stablehlo.transpose %d0, dims = [1, 0, 2] : (tensor<3x8x4xf64>) -> tensor<8x3x4xf64>
  %t1 = stablehlo.transpose %d1, dims = [1, 0, 2] : (tensor<3x8x4xf64>) -> tensor<8x3x4xf64>
  %c = stablehlo.concatenate %t0, %t1, dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>) -> tensor<8x6x4xf64>
  return %c : tensor<8x6x4xf64>
}

// CHECK-LABEL: func.func @stack_through_transpose(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x4x4xf64>) -> tensor<8x6x4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 0 : (tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>) -> tensor<2x8x4x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.dot_general %[[V0]], %[[V5]], contracting_dims = [1] x [3] : (tensor<3x4xf64>, tensor<2x8x4x4xf64>) -> tensor<3x2x8x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.transpose %[[V6]], dims = [1, 0, 2, 3] : (tensor<3x2x8x4xf64>) -> tensor<2x3x8x4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<2x3x8x4xf64>) -> tensor<6x8x4xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.transpose %[[V8]], dims = [1, 0, 2] : (tensor<6x8x4xf64>) -> tensor<8x6x4xf64>
// CHECK-NEXT: return %[[V9]] : tensor<8x6x4xf64>
// CHECK-NEXT: }

// Each of two lhs against each of three rhs, every piece given a leading unit
// dimension and concatenated along it, rhs-major (as BK2PARTIAL's basis rows
// against its fluxes): each run sharing an rhs merges into one dot of the
// stacked lhs, then the runs into one dot of the stacked lhs against the
// stacked rhs.
func.func @grid_rhs_major(%l0: tensor<2x4xf64>, %l1: tensor<2x4xf64>, %r0: tensor<5x4xf64>, %r1: tensor<5x4xf64>, %r2: tensor<5x4xf64>) -> tensor<6x5x2xf64> {
  %d0 = stablehlo.dot_general %l0, %r0, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p0 = stablehlo.broadcast_in_dim %d0, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d1 = stablehlo.dot_general %l1, %r0, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p1 = stablehlo.broadcast_in_dim %d1, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d2 = stablehlo.dot_general %l0, %r1, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p2 = stablehlo.broadcast_in_dim %d2, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d3 = stablehlo.dot_general %l1, %r1, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p3 = stablehlo.broadcast_in_dim %d3, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d4 = stablehlo.dot_general %l0, %r2, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p4 = stablehlo.broadcast_in_dim %d4, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d5 = stablehlo.dot_general %l1, %r2, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p5 = stablehlo.broadcast_in_dim %d5, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %c = stablehlo.concatenate %p0, %p1, %p2, %p3, %p4, %p5, dim = 0 : (tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>) -> tensor<6x5x2xf64>
  return %c : tensor<6x5x2xf64>
}

// CHECK-LABEL: func.func @grid_rhs_major(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x4xf64>, %[[V1:[^ ,:)]+]]: tensor<2x4xf64>, %[[V2:[^ ,:)]+]]: tensor<5x4xf64>, %[[V3:[^ ,:)]+]]: tensor<5x4xf64>, %[[V4:[^ ,:)]+]]: tensor<5x4xf64>) -> tensor<6x5x2xf64> {
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.concatenate %[[V5]], %[[V6]], dim = 0 : (tensor<1x2x4xf64>, tensor<1x2x4xf64>) -> tensor<2x2x4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.concatenate %[[V8]], %[[V9]], %[[V10]], dim = 0 : (tensor<1x5x4xf64>, tensor<1x5x4xf64>, tensor<1x5x4xf64>) -> tensor<3x5x4xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.dot_general %[[V7]], %[[V11]], contracting_dims = [2] x [2] : (tensor<2x2x4xf64>, tensor<3x5x4xf64>) -> tensor<2x2x3x5xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.transpose %[[V12]], dims = [2, 0, 1, 3] : (tensor<2x2x3x5xf64>) -> tensor<3x2x2x5xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reshape %[[V13]] : (tensor<3x2x2x5xf64>) -> tensor<6x2x5xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V14]], dims = [0, 2, 1] : (tensor<6x2x5xf64>) -> tensor<6x5x2xf64>
// CHECK-NEXT: return %[[V15]] : tensor<6x5x2xf64>
// CHECK-NEXT: }

// The same grid, lhs-major.
func.func @grid_lhs_major(%l0: tensor<2x4xf64>, %l1: tensor<2x4xf64>, %r0: tensor<5x4xf64>, %r1: tensor<5x4xf64>, %r2: tensor<5x4xf64>) -> tensor<6x5x2xf64> {
  %d0 = stablehlo.dot_general %l0, %r0, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p0 = stablehlo.broadcast_in_dim %d0, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d1 = stablehlo.dot_general %l0, %r1, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p1 = stablehlo.broadcast_in_dim %d1, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d2 = stablehlo.dot_general %l0, %r2, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p2 = stablehlo.broadcast_in_dim %d2, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d3 = stablehlo.dot_general %l1, %r0, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p3 = stablehlo.broadcast_in_dim %d3, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d4 = stablehlo.dot_general %l1, %r1, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p4 = stablehlo.broadcast_in_dim %d4, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %d5 = stablehlo.dot_general %l1, %r2, contracting_dims = [1] x [1] : (tensor<2x4xf64>, tensor<5x4xf64>) -> tensor<2x5xf64>
  %p5 = stablehlo.broadcast_in_dim %d5, dims = [2, 1] : (tensor<2x5xf64>) -> tensor<1x5x2xf64>
  %c = stablehlo.concatenate %p0, %p1, %p2, %p3, %p4, %p5, dim = 0 : (tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>, tensor<1x5x2xf64>) -> tensor<6x5x2xf64>
  return %c : tensor<6x5x2xf64>
}

// CHECK-LABEL: func.func @grid_lhs_major(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x4xf64>, %[[V1:[^ ,:)]+]]: tensor<2x4xf64>, %[[V2:[^ ,:)]+]]: tensor<5x4xf64>, %[[V3:[^ ,:)]+]]: tensor<5x4xf64>, %[[V4:[^ ,:)]+]]: tensor<5x4xf64>) -> tensor<6x5x2xf64> {
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.concatenate %[[V5]], %[[V6]], %[[V7]], dim = 0 : (tensor<1x5x4xf64>, tensor<1x5x4xf64>, tensor<1x5x4xf64>) -> tensor<3x5x4xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.concatenate %[[V9]], %[[V10]], dim = 0 : (tensor<1x2x4xf64>, tensor<1x2x4xf64>) -> tensor<2x2x4xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.dot_general %[[V11]], %[[V8]], contracting_dims = [2] x [2] : (tensor<2x2x4xf64>, tensor<3x5x4xf64>) -> tensor<2x2x3x5xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.transpose %[[V12]], dims = [1, 0, 2, 3] : (tensor<2x2x3x5xf64>) -> tensor<2x2x3x5xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reshape %[[V13]] : (tensor<2x2x3x5xf64>) -> tensor<2x6x5xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.transpose %[[V14]], dims = [1, 0, 2] : (tensor<2x6x5xf64>) -> tensor<6x2x5xf64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V15]], dims = [0, 2, 1] : (tensor<6x2x5xf64>) -> tensor<6x5x2xf64>
// CHECK-NEXT: return %[[V16]] : tensor<6x5x2xf64>
// CHECK-NEXT: }

// Only the first two operands merge; the concatenate keeps the third.
func.func @partial_run(%a0: tensor<3x4xf64>, %a1: tensor<2x4xf64>, %b: tensor<4x6xf64>, %z: tensor<1x6xf64>) -> tensor<6x6xf64> {
  %d0 = stablehlo.dot_general %a0, %b, contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4x6xf64>) -> tensor<3x6xf64>
  %d1 = stablehlo.dot_general %a1, %b, contracting_dims = [1] x [0] : (tensor<2x4xf64>, tensor<4x6xf64>) -> tensor<2x6xf64>
  %c = stablehlo.concatenate %d0, %d1, %z, dim = 0 : (tensor<3x6xf64>, tensor<2x6xf64>, tensor<1x6xf64>) -> tensor<6x6xf64>
  return %c : tensor<6x6xf64>
}

// CHECK-LABEL: func.func @partial_run(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<2x4xf64>, %[[V2:[^ ,:)]+]]: tensor<4x6xf64>, %[[V3:[^ ,:)]+]]: tensor<1x6xf64>) -> tensor<6x6xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<3x4xf64>, tensor<2x4xf64>) -> tensor<5x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dot_general %[[V4]], %[[V2]], contracting_dims = [1] x [0] : (tensor<5x4xf64>, tensor<4x6xf64>) -> tensor<5x6xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.concatenate %[[V5]], %[[V3]], dim = 0 : (tensor<5x6xf64>, tensor<1x6xf64>) -> tensor<6x6xf64>
// CHECK-NEXT: return %[[V6]] : tensor<6x6xf64>
// CHECK-NEXT: }

// Pieces reshaped so that the concatenated dimension holds whole rows of
// their dimension 1: one dot of the concatenated %x along it, reshaped.
func.func @regrouped(%x0: tensor<8x3x3x3xf64>, %x1: tensor<8x3x3x3xf64>, %x2: tensor<8x3x3x3xf64>, %b: tensor<3x3xf64>) -> tensor<8x27x3xf64> {
  %d0 = stablehlo.dot_general %x0, %b, contracting_dims = [2] x [1] : (tensor<8x3x3x3xf64>, tensor<3x3xf64>) -> tensor<8x3x3x3xf64>
  %d1 = stablehlo.dot_general %x1, %b, contracting_dims = [2] x [1] : (tensor<8x3x3x3xf64>, tensor<3x3xf64>) -> tensor<8x3x3x3xf64>
  %d2 = stablehlo.dot_general %x2, %b, contracting_dims = [2] x [1] : (tensor<8x3x3x3xf64>, tensor<3x3xf64>) -> tensor<8x3x3x3xf64>
  %r0 = stablehlo.reshape %d0 : (tensor<8x3x3x3xf64>) -> tensor<8x9x3xf64>
  %r1 = stablehlo.reshape %d1 : (tensor<8x3x3x3xf64>) -> tensor<8x9x3xf64>
  %r2 = stablehlo.reshape %d2 : (tensor<8x3x3x3xf64>) -> tensor<8x9x3xf64>
  %c = stablehlo.concatenate %r0, %r1, %r2, dim = 1 : (tensor<8x9x3xf64>, tensor<8x9x3xf64>, tensor<8x9x3xf64>) -> tensor<8x27x3xf64>
  return %c : tensor<8x27x3xf64>
}

// CHECK-LABEL: func.func @regrouped(
// CHECK-SAME: %[[X0:[^ ,:)]+]]: tensor<8x3x3x3xf64>, %[[X1:[^ ,:)]+]]: tensor<8x3x3x3xf64>, %[[X2:[^ ,:)]+]]: tensor<8x3x3x3xf64>, %[[B:[^ ,:)]+]]: tensor<3x3xf64>) -> tensor<8x27x3xf64> {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[X0]], %[[X1]], %[[X2]], dim = 1 : (tensor<8x3x3x3xf64>, tensor<8x3x3x3xf64>, tensor<8x3x3x3xf64>) -> tensor<8x9x3x3xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[C]], %[[B]], contracting_dims = [2] x [1] : (tensor<8x9x3x3xf64>, tensor<3x3xf64>) -> tensor<8x9x3x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[D]] : (tensor<8x9x3x3xf64>) -> tensor<8x27x3xf64>
// CHECK-NEXT: return %[[R]] : tensor<8x27x3xf64>
// CHECK-NEXT: }

// A reshape that only inserts the concatenated dimension: left alone here.
func.func @inserted(%x0: tensor<8x3xf64>, %x1: tensor<8x3xf64>, %b: tensor<4x3xf64>) -> tensor<2x8x4xf64> {
  %d0 = stablehlo.dot_general %x0, %b, contracting_dims = [1] x [1] : (tensor<8x3xf64>, tensor<4x3xf64>) -> tensor<8x4xf64>
  %d1 = stablehlo.dot_general %x1, %b, contracting_dims = [1] x [1] : (tensor<8x3xf64>, tensor<4x3xf64>) -> tensor<8x4xf64>
  %r0 = stablehlo.reshape %d0 : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
  %r1 = stablehlo.reshape %d1 : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
  %c = stablehlo.concatenate %r0, %r1, dim = 0 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<2x8x4xf64>
  return %c : tensor<2x8x4xf64>
}

// CHECK-LABEL: func.func @inserted(
// CHECK-SAME: %[[X0:[^ ,:)]+]]: tensor<8x3xf64>, %[[X1:[^ ,:)]+]]: tensor<8x3xf64>, %[[B:[^ ,:)]+]]: tensor<4x3xf64>) -> tensor<2x8x4xf64> {
// CHECK-NEXT: %[[D0:[^ ,:)]+]] = stablehlo.dot_general %[[X0]], %[[B]], contracting_dims = [1] x [1] : (tensor<8x3xf64>, tensor<4x3xf64>) -> tensor<8x4xf64>
// CHECK-NEXT: %[[D1:[^ ,:)]+]] = stablehlo.dot_general %[[X1]], %[[B]], contracting_dims = [1] x [1] : (tensor<8x3xf64>, tensor<4x3xf64>) -> tensor<8x4xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reshape %[[D0]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[R1:[^ ,:)]+]] = stablehlo.reshape %[[D1]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 0 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<2x8x4xf64>
// CHECK-NEXT: return %[[C]] : tensor<2x8x4xf64>
// CHECK-NEXT: }

// Pieces of the same contraction whose shared operand is reached through
// different views (a transposed layout, a unit-dimension reshape of a slice
// against a slice of a reshape): one dot of the stacked varying operands
// against the first piece's operand.
func.func @views(%x0: tensor<8x4xf64>, %x1: tensor<8x4xf64>, %b: tensor<6x4xf64>) -> tensor<2x8x5xf64> {
  %b0 = stablehlo.slice %b [0:5, 0:4] : (tensor<6x4xf64>) -> tensor<5x4xf64>
  %d0 = stablehlo.dot_general %x0, %b0, contracting_dims = [1] x [1] : (tensor<8x4xf64>, tensor<5x4xf64>) -> tensor<8x5xf64>
  %bt = stablehlo.transpose %b, dims = [1, 0] : (tensor<6x4xf64>) -> tensor<4x6xf64>
  %b1 = stablehlo.slice %bt [0:4, 0:5] : (tensor<4x6xf64>) -> tensor<4x5xf64>
  %d1 = stablehlo.dot_general %x1, %b1, contracting_dims = [1] x [0] : (tensor<8x4xf64>, tensor<4x5xf64>) -> tensor<8x5xf64>
  %r0 = stablehlo.broadcast_in_dim %d0, dims = [1, 2] : (tensor<8x5xf64>) -> tensor<1x8x5xf64>
  %r1 = stablehlo.broadcast_in_dim %d1, dims = [1, 2] : (tensor<8x5xf64>) -> tensor<1x8x5xf64>
  %c = stablehlo.concatenate %r0, %r1, dim = 0 : (tensor<1x8x5xf64>, tensor<1x8x5xf64>) -> tensor<2x8x5xf64>
  return %c : tensor<2x8x5xf64>
}

// CHECK-LABEL: func.func @views(
// CHECK-SAME: %[[X0:[^ ,:)]+]]: tensor<8x4xf64>, %[[X1:[^ ,:)]+]]: tensor<8x4xf64>, %[[B:[^ ,:)]+]]: tensor<6x4xf64>) -> tensor<2x8x5xf64> {
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.slice %[[B]] [0:5, 0:4] : (tensor<6x4xf64>) -> tensor<5x4xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reshape %[[X0]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[R1:[^ ,:)]+]] = stablehlo.reshape %[[X1]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 0 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<2x8x4xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[S]], %[[B0]], contracting_dims = [2] x [1] : (tensor<2x8x4xf64>, tensor<5x4xf64>) -> tensor<2x8x5xf64>
// CHECK-NEXT: return %[[D]] : tensor<2x8x5xf64>
// CHECK-NEXT: }

// The operands in the other order, each result already in the same layout
// (both free groups have the same extent).
func.func @swapped(%x0: tensor<8x4xf64>, %x1: tensor<8x4xf64>, %b: tensor<8x4xf64>) -> tensor<2x8x8xf64> {
  %d0 = stablehlo.dot_general %x0, %b, contracting_dims = [1] x [1] : (tensor<8x4xf64>, tensor<8x4xf64>) -> tensor<8x8xf64>
  %d1 = stablehlo.dot_general %b, %x1, contracting_dims = [1] x [1] : (tensor<8x4xf64>, tensor<8x4xf64>) -> tensor<8x8xf64>
  %r0 = stablehlo.broadcast_in_dim %d0, dims = [1, 2] : (tensor<8x8xf64>) -> tensor<1x8x8xf64>
  %r1 = stablehlo.broadcast_in_dim %d1, dims = [1, 2] : (tensor<8x8xf64>) -> tensor<1x8x8xf64>
  %c = stablehlo.concatenate %r0, %r1, dim = 0 : (tensor<1x8x8xf64>, tensor<1x8x8xf64>) -> tensor<2x8x8xf64>
  return %c : tensor<2x8x8xf64>
}

// CHECK-LABEL: func.func @swapped(
// CHECK-SAME: %[[X0:[^ ,:)]+]]: tensor<8x4xf64>, %[[X1:[^ ,:)]+]]: tensor<8x4xf64>, %[[B:[^ ,:)]+]]: tensor<8x4xf64>) -> tensor<2x8x8xf64> {
// CHECK-NEXT: %[[L0:[^ ,:)]+]] = stablehlo.reshape %[[X0]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[L1:[^ ,:)]+]] = stablehlo.reshape %[[B]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[L:[^ ,:)]+]] = stablehlo.concatenate %[[L0]], %[[L1]], dim = 0 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<2x8x4xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reshape %[[B]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[R1:[^ ,:)]+]] = stablehlo.reshape %[[X1]] : (tensor<8x4xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 0 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<2x8x4xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[L]], %[[R]], batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<2x8x4xf64>, tensor<2x8x4xf64>) -> tensor<2x8x8xf64>
// CHECK-NEXT: return %[[D]] : tensor<2x8x8xf64>
// CHECK-NEXT: }

// A run that mixes a piece carrying the concatenated dimension as a free
// dimension of its varying operand (an earlier merge of such a run) with a
// piece whose views insert it as a unit dimension: the pieces' varying
// operands, a unit dimension inserted for the single piece, concatenate
// along it and one dot replaces the run, with the carrier's views.
func.func @mixed(%b: tensor<5x6x6x8xf64>, %y0: tensor<6x5x8x6xf64>, %y12: tensor<2x6x5x8x6xf64>) -> tensor<3x8x6x6x6xf64> {
  %d0 = stablehlo.dot_general %b, %y0, batching_dims = [3, 1] x [2, 3], contracting_dims = [0] x [1] : (tensor<5x6x6x8xf64>, tensor<6x5x8x6xf64>) -> tensor<8x6x6x6xf64>
  %p0 = stablehlo.broadcast_in_dim %d0, dims = [1, 2, 4, 3] : (tensor<8x6x6x6xf64>) -> tensor<1x8x6x6x6xf64>
  %d12 = stablehlo.dot_general %b, %y12, batching_dims = [3, 1] x [3, 4], contracting_dims = [0] x [2] : (tensor<5x6x6x8xf64>, tensor<2x6x5x8x6xf64>) -> tensor<8x6x6x2x6xf64>
  %p12 = stablehlo.transpose %d12, dims = [3, 0, 1, 4, 2] : (tensor<8x6x6x2x6xf64>) -> tensor<2x8x6x6x6xf64>
  %c = stablehlo.concatenate %p0, %p12, dim = 0 : (tensor<1x8x6x6x6xf64>, tensor<2x8x6x6x6xf64>) -> tensor<3x8x6x6x6xf64>
  return %c : tensor<3x8x6x6x6xf64>
}

// CHECK-LABEL: func.func @mixed(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<5x6x6x8xf64>, %[[Y0:[^ ,:)]+]]: tensor<6x5x8x6xf64>, %[[Y12:[^ ,:)]+]]: tensor<2x6x5x8x6xf64>) -> tensor<3x8x6x6x6xf64> {
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.reshape %[[Y0]] : (tensor<6x5x8x6xf64>) -> tensor<1x6x5x8x6xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[U]], %[[Y12]], dim = 0 : (tensor<1x6x5x8x6xf64>, tensor<2x6x5x8x6xf64>) -> tensor<3x6x5x8x6xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[B]], %[[S]], batching_dims = [3, 1] x [3, 4], contracting_dims = [0] x [2] : (tensor<5x6x6x8xf64>, tensor<3x6x5x8x6xf64>) -> tensor<8x6x6x3x6xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[D]], dims = [3, 0, 1, 4, 2] : (tensor<8x6x6x3x6xf64>) -> tensor<3x8x6x6x6xf64>
// CHECK-NEXT: return %[[T]] : tensor<3x8x6x6x6xf64>
// CHECK-NEXT: }

// The single piece's varying operand in another layout than the carrier's:
// transposed into it first.
func.func @mixed_layout(%b: tensor<5x6xf64>, %y0: tensor<5x8x6x5xf64>, %y12: tensor<2x6x5x8x5xf64>) -> tensor<3x6x5x8x6xf64> {
  %d0 = stablehlo.dot_general %b, %y0, contracting_dims = [0] x [3] : (tensor<5x6xf64>, tensor<5x8x6x5xf64>) -> tensor<6x5x8x6xf64>
  %p0 = stablehlo.reshape %d0 : (tensor<6x5x8x6xf64>) -> tensor<1x6x5x8x6xf64>
  %d12 = stablehlo.dot_general %b, %y12, contracting_dims = [0] x [4] : (tensor<5x6xf64>, tensor<2x6x5x8x5xf64>) -> tensor<6x2x6x5x8xf64>
  %p12 = stablehlo.transpose %d12, dims = [1, 0, 3, 4, 2] : (tensor<6x2x6x5x8xf64>) -> tensor<2x6x5x8x6xf64>
  %c = stablehlo.concatenate %p0, %p12, dim = 0 : (tensor<1x6x5x8x6xf64>, tensor<2x6x5x8x6xf64>) -> tensor<3x6x5x8x6xf64>
  return %c : tensor<3x6x5x8x6xf64>
}

// CHECK-LABEL: func.func @mixed_layout(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<5x6xf64>, %[[Y0:[^ ,:)]+]]: tensor<5x8x6x5xf64>, %[[Y12:[^ ,:)]+]]: tensor<2x6x5x8x5xf64>) -> tensor<3x6x5x8x6xf64> {
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.transpose %[[Y0]], dims = [2, 0, 1, 3] : (tensor<5x8x6x5xf64>) -> tensor<6x5x8x5xf64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.reshape %[[P]] : (tensor<6x5x8x5xf64>) -> tensor<1x6x5x8x5xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[U]], %[[Y12]], dim = 0 : (tensor<1x6x5x8x5xf64>, tensor<2x6x5x8x5xf64>) -> tensor<3x6x5x8x5xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[B]], %[[S]], contracting_dims = [0] x [4] : (tensor<5x6xf64>, tensor<3x6x5x8x5xf64>) -> tensor<6x3x6x5x8xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[D]], dims = [1, 0, 3, 4, 2] : (tensor<6x3x6x5x8xf64>) -> tensor<3x6x5x8x6xf64>
// CHECK-NEXT: return %[[T]] : tensor<3x6x5x8x6xf64>
// CHECK-NEXT: }

// A grouped carrier: the run of two is merged with the contraction's
// free dimension into the concatenated one ((2, 3) -> 6) by a reshape
// among its views, so the single piece (a whole group of 3) is one
// window of the varying operand, concatenated at the input.
func.func @grouped(%b: tensor<3x3xf64>, %x0: tensor<8x3x3x3xf64>, %x12: tensor<2x8x3x3x3xf64>) -> tensor<8x9x3x3xf64> {
  %d0 = stablehlo.dot_general %b, %x0, contracting_dims = [1] x [1] : (tensor<3x3xf64>, tensor<8x3x3x3xf64>) -> tensor<3x8x3x3xf64>
  %p0 = stablehlo.transpose %d0, dims = [1, 0, 2, 3] : (tensor<3x8x3x3xf64>) -> tensor<8x3x3x3xf64>
  %d12 = stablehlo.dot_general %b, %x12, contracting_dims = [1] x [2] : (tensor<3x3xf64>, tensor<2x8x3x3x3xf64>) -> tensor<3x2x8x3x3xf64>
  %t12 = stablehlo.transpose %d12, dims = [1, 0, 2, 3, 4] : (tensor<3x2x8x3x3xf64>) -> tensor<2x3x8x3x3xf64>
  %r12 = stablehlo.reshape %t12 : (tensor<2x3x8x3x3xf64>) -> tensor<6x8x3x3xf64>
  %p12 = stablehlo.transpose %r12, dims = [1, 0, 2, 3] : (tensor<6x8x3x3xf64>) -> tensor<8x6x3x3xf64>
  %c = stablehlo.concatenate %p0, %p12, dim = 1 : (tensor<8x3x3x3xf64>, tensor<8x6x3x3xf64>) -> tensor<8x9x3x3xf64>
  return %c : tensor<8x9x3x3xf64>
}

// CHECK-LABEL: func.func @grouped(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x3xf64>, %[[X0:[^ ,:)]+]]: tensor<8x3x3x3xf64>, %[[X12:[^ ,:)]+]]: tensor<2x8x3x3x3xf64>) -> tensor<8x9x3x3xf64> {
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.reshape %[[X0]] : (tensor<8x3x3x3xf64>) -> tensor<1x8x3x3x3xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[U]], %[[X12]], dim = 0 : (tensor<1x8x3x3x3xf64>, tensor<2x8x3x3x3xf64>) -> tensor<3x8x3x3x3xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[B]], %[[C]], contracting_dims = [1] x [2] : (tensor<3x3xf64>, tensor<3x8x3x3x3xf64>) -> tensor<3x3x8x3x3xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[D]], dims = [1, 0, 2, 3, 4] : (tensor<3x3x8x3x3xf64>) -> tensor<3x3x8x3x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[T]] : (tensor<3x3x8x3x3xf64>) -> tensor<9x8x3x3xf64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 0, 2, 3] : (tensor<9x8x3x3xf64>) -> tensor<8x9x3x3xf64>
// CHECK-NEXT: return %[[P]] : tensor<8x9x3x3xf64>
// CHECK-NEXT: }

// A piece spelled with its operands swapped, its transpose undoing that:
// every dimension of its result in the first's form lands where the
// first's does, so it joins the run (the shared operand's free dimension
// is the concatenated one; the varying operands stack).
func.func @swapped_spelling(%b: tensor<3x4xf64>, %x0: tensor<8x4x4xf64>, %x1: tensor<8x4x4xf64>, %x2: tensor<8x4x4xf64>) -> tensor<8x9x4xf64> {
  %d0 = stablehlo.dot_general %b, %x0, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4x4xf64>) -> tensor<3x8x4xf64>
  %t0 = stablehlo.transpose %d0, dims = [1, 0, 2] : (tensor<3x8x4xf64>) -> tensor<8x3x4xf64>
  %d1 = stablehlo.dot_general %x1, %b, contracting_dims = [1] x [1] : (tensor<8x4x4xf64>, tensor<3x4xf64>) -> tensor<8x4x3xf64>
  %t1 = stablehlo.transpose %d1, dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4xf64>
  %d2 = stablehlo.dot_general %b, %x2, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4x4xf64>) -> tensor<3x8x4xf64>
  %t2 = stablehlo.transpose %d2, dims = [1, 0, 2] : (tensor<3x8x4xf64>) -> tensor<8x3x4xf64>
  %c = stablehlo.concatenate %t0, %t1, %t2, dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>, tensor<8x3x4xf64>) -> tensor<8x9x4xf64>
  return %c : tensor<8x9x4xf64>
}

// CHECK-LABEL: func.func @swapped_spelling(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X0:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[X1:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[X2:[^ ,:)]+]]: tensor<8x4x4xf64>) -> tensor<8x9x4xf64> {
// CHECK-NEXT: %[[U0:[^ ,:)]+]] = stablehlo.reshape %[[X0]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[U1:[^ ,:)]+]] = stablehlo.reshape %[[X1]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[U2:[^ ,:)]+]] = stablehlo.reshape %[[X2]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[U0]], %[[U1]], %[[U2]], dim = 0 : (tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[B]], %[[S]], contracting_dims = [1] x [2] : (tensor<3x4xf64>, tensor<3x8x4x4xf64>) -> tensor<3x3x8x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[D]], dims = [1, 0, 2, 3] : (tensor<3x3x8x4xf64>) -> tensor<3x3x8x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[T]] : (tensor<3x3x8x4xf64>) -> tensor<9x8x4xf64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 0, 2] : (tensor<9x8x4xf64>) -> tensor<8x9x4xf64>
// CHECK-NEXT: return %[[P]] : tensor<8x9x4xf64>
// CHECK-NEXT: }

// A swapped piece of a different extent along the concatenated dimension
// (the run of 65 rows beside two single rows, as the vector mass apply has
// them): the merged result's extent is summed through each piece's own
// result dimension.
func.func @swapped_extent(%b: tensor<5x6xf64>, %x0: tensor<8x5x6xf64>, %x1: tensor<8x5x6xf64>, %xr: tensor<8x65x6xf64>) -> tensor<8x75x5xf64> {
  %d0 = stablehlo.dot_general %b, %x0, contracting_dims = [1] x [2] : (tensor<5x6xf64>, tensor<8x5x6xf64>) -> tensor<5x8x5xf64>
  %t0 = stablehlo.transpose %d0, dims = [1, 2, 0] : (tensor<5x8x5xf64>) -> tensor<8x5x5xf64>
  %d1 = stablehlo.dot_general %b, %x1, contracting_dims = [1] x [2] : (tensor<5x6xf64>, tensor<8x5x6xf64>) -> tensor<5x8x5xf64>
  %t1 = stablehlo.transpose %d1, dims = [1, 2, 0] : (tensor<5x8x5xf64>) -> tensor<8x5x5xf64>
  %dr = stablehlo.dot_general %xr, %b, contracting_dims = [2] x [1] : (tensor<8x65x6xf64>, tensor<5x6xf64>) -> tensor<8x65x5xf64>
  %c = stablehlo.concatenate %t0, %t1, %dr, dim = 1 : (tensor<8x5x5xf64>, tensor<8x5x5xf64>, tensor<8x65x5xf64>) -> tensor<8x75x5xf64>
  return %c : tensor<8x75x5xf64>
}

// CHECK-LABEL: func.func @swapped_extent(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<5x6xf64>, %[[X0:[^ ,:)]+]]: tensor<8x5x6xf64>, %[[X1:[^ ,:)]+]]: tensor<8x5x6xf64>, %[[XR:[^ ,:)]+]]: tensor<8x65x6xf64>) -> tensor<8x75x5xf64> {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[X0]], %[[X1]], %[[XR]], dim = 1 : (tensor<8x5x6xf64>, tensor<8x5x6xf64>, tensor<8x65x6xf64>) -> tensor<8x75x6xf64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.dot_general %[[B]], %[[C]], contracting_dims = [1] x [2] : (tensor<5x6xf64>, tensor<8x75x6xf64>) -> tensor<5x8x75xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[D]], dims = [1, 2, 0] : (tensor<5x8x75xf64>) -> tensor<8x75x5xf64>
// CHECK-NEXT: return %[[T]] : tensor<8x75x5xf64>
// CHECK-NEXT: }
