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
