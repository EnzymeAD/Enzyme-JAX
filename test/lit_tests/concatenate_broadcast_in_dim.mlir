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
// an inserted one, and the expanded unit dimensions are dropped.
func.func @expanded_unit(%s0: tensor<1x1xf64>, %s1: tensor<1x1xf64>, %s2: tensor<1x1xf64>) -> tensor<8x15x5xf64> {
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x5x5xf64>
  %b1 = stablehlo.broadcast_in_dim %s1, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x5x5xf64>
  %b2 = stablehlo.broadcast_in_dim %s2, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x5x5xf64>
  %c = stablehlo.concatenate %b0, %b1, %b2, dim = 1 : (tensor<8x5x5xf64>, tensor<8x5x5xf64>, tensor<8x5x5xf64>) -> tensor<8x15x5xf64>
  return %c : tensor<8x15x5xf64>
}

// CHECK-LABEL: func.func @expanded_unit(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1x1xf64>, %[[V1:[^ ,:)]+]]: tensor<1x1xf64>, %[[V2:[^ ,:)]+]]: tensor<1x1xf64>) -> tensor<8x15x5xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], %[[V5]], dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V6]], dims = [1] : (tensor<3xf64>) -> tensor<8x3x5x5xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<8x3x5x5xf64>) -> tensor<8x15x5xf64>
// CHECK-NEXT: return %[[V8]] : tensor<8x15x5xf64>
// CHECK-NEXT: }

// The broadcasts of one family read the same values through different
// operand layouts (a reduction over a differently placed dimension in each:
// transposed, or with a unit dimension the broadcast expands): each is
// brought to the first's layout by a view and they concatenate there.
func.func @layouts(%a: tensor<8x3x4xf64>, %b: tensor<8x4x3xf64>, %c: tensor<8x4x3x1xf64>) -> tensor<8x4x9x3xf64> {
  %0 = stablehlo.broadcast_in_dim %a, dims = [0, 2, 1] : (tensor<8x3x4xf64>) -> tensor<8x4x3x3xf64>
  %1 = stablehlo.broadcast_in_dim %b, dims = [0, 1, 2] : (tensor<8x4x3xf64>) -> tensor<8x4x3x3xf64>
  %2 = stablehlo.broadcast_in_dim %c, dims = [0, 1, 2, 3] : (tensor<8x4x3x1xf64>) -> tensor<8x4x3x3xf64>
  %3 = stablehlo.concatenate %0, %1, %2, dim = 2 : (tensor<8x4x3x3xf64>, tensor<8x4x3x3xf64>, tensor<8x4x3x3xf64>) -> tensor<8x4x9x3xf64>
  return %3 : tensor<8x4x9x3xf64>
}

// CHECK-LABEL: func.func @layouts(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[B:[^ ,:)]+]]: tensor<8x4x3xf64>, %[[C:[^ ,:)]+]]: tensor<8x4x3x1xf64>) -> tensor<8x4x9x3xf64> {
// CHECK-NEXT: %[[BT:[^ ,:)]+]] = stablehlo.transpose %[[B]], dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %[[CR:[^ ,:)]+]] = stablehlo.reshape %[[C]] : (tensor<8x4x3x1xf64>) -> tensor<8x4x3xf64>
// CHECK-NEXT: %[[CT:[^ ,:)]+]] = stablehlo.transpose %[[CR]], dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %[[CC:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[BT]], %[[CT]], dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>, tensor<8x3x4xf64>) -> tensor<8x9x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[CC]], dims = [0, 2, 1] : (tensor<8x9x4xf64>) -> tensor<8x4x9x3xf64>
// CHECK-NEXT: return %[[R]] : tensor<8x4x9x3xf64>
// CHECK-NEXT: }

// Operands differing only in unit dimensions that mark output dimensions
// (one carries the expanded dimension and a trailing unit one as size-1
// dimensions, the others not): the same layout.
func.func @unit_dims(%a: tensor<1x8x4x4x1xf64>, %b: tensor<8x4x4xf64>, %c: tensor<8x4x4xf64>) -> tensor<9x8x4x4x1xf64> {
  %0 = stablehlo.broadcast_in_dim %a, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x4x1xf64>) -> tensor<3x8x4x4x1xf64>
  %1 = stablehlo.broadcast_in_dim %b, dims = [1, 2, 3] : (tensor<8x4x4xf64>) -> tensor<3x8x4x4x1xf64>
  %2 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<8x4x4xf64>) -> tensor<3x8x4x4x1xf64>
  %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<3x8x4x4x1xf64>, tensor<3x8x4x4x1xf64>, tensor<3x8x4x4x1xf64>) -> tensor<9x8x4x4x1xf64>
  return %3 : tensor<9x8x4x4x1xf64>
}

// CHECK-LABEL: func.func @unit_dims(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<1x8x4x4x1xf64>, %[[B:[^ ,:)]+]]: tensor<8x4x4xf64>, %[[C:[^ ,:)]+]]: tensor<8x4x4xf64>) -> tensor<9x8x4x4x1xf64> {
// CHECK-NEXT: %[[UA:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<1x8x4x4x1xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[UB:[^ ,:)]+]] = stablehlo.reshape %[[B]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[UC:[^ ,:)]+]] = stablehlo.reshape %[[C]] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[UA]], %[[UB]], %[[UC]], dim = 0 : (tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[BC:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S]], dims = [0, 2, 3, 4] : (tensor<3x8x4x4xf64>) -> tensor<3x3x8x4x4x1xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[BC]] : (tensor<3x3x8x4x4x1xf64>) -> tensor<9x8x4x4x1xf64>
// CHECK-NEXT: return %[[R]] : tensor<9x8x4x4x1xf64>
// CHECK-NEXT: }

// Broadcasts expanding the concatenated dimension to different extents (a
// selection of basis rows: 5 of B, then 20 of G, twice): the concatenation
// is the broadcast of the same concatenation without the dimension every
// operand expands (the elements), as much smaller.
func.func @hoist_expanded(%b: tensor<5x5xf64>, %g: tensor<5x5xf64>) -> tensor<5x8x50x5xf64> {
  %0 = stablehlo.broadcast_in_dim %b, dims = [3, 0] : (tensor<5x5xf64>) -> tensor<5x8x5x5xf64>
  %1 = stablehlo.broadcast_in_dim %g, dims = [3, 0] : (tensor<5x5xf64>) -> tensor<5x8x20x5xf64>
  %2 = stablehlo.concatenate %0, %1, %0, %1, dim = 2 : (tensor<5x8x5x5xf64>, tensor<5x8x20x5xf64>, tensor<5x8x5x5xf64>, tensor<5x8x20x5xf64>) -> tensor<5x8x50x5xf64>
  return %2 : tensor<5x8x50x5xf64>
}

// CHECK-LABEL: func.func @hoist_expanded(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<5x5xf64>, %[[G:[^ ,:)]+]]: tensor<5x5xf64>) -> tensor<5x8x50x5xf64> {
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [2, 0] : (tensor<5x5xf64>) -> tensor<5x5x5xf64>
// CHECK-NEXT: %[[G0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[G]], dims = [2, 0] : (tensor<5x5xf64>) -> tensor<5x20x5xf64>
// CHECK-NEXT: %[[B1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [2, 0] : (tensor<5x5xf64>) -> tensor<5x5x5xf64>
// CHECK-NEXT: %[[G1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[G]], dims = [2, 0] : (tensor<5x5xf64>) -> tensor<5x20x5xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[B0]], %[[G0]], %[[B1]], %[[G1]], dim = 1 : (tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>, tensor<5x20x5xf64>) -> tensor<5x50x5xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[C]], dims = [0, 2, 3] : (tensor<5x50x5xf64>) -> tensor<5x8x50x5xf64>
// CHECK-NEXT: return %[[R]] : tensor<5x8x50x5xf64>
// CHECK-NEXT: }

// Hoisted with an operand laid out in order but carrying a unit dimension the
// broadcast expands: it is reshaped to the kept dimensions before it is
// broadcast to the inner concatenation.
func.func @hoist_expanded_unit(%b: tensor<3x1xf64>, %g: tensor<3x1xf64>) -> tensor<4x27x3xf64> {
  %0 = stablehlo.broadcast_in_dim %b, dims = [2, 0] : (tensor<3x1xf64>) -> tensor<4x5x3xf64>
  %1 = stablehlo.broadcast_in_dim %g, dims = [2, 0] : (tensor<3x1xf64>) -> tensor<4x22x3xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x5x3xf64>, tensor<4x22x3xf64>) -> tensor<4x27x3xf64>
  return %2 : tensor<4x27x3xf64>
}

// CHECK-LABEL: func.func @hoist_expanded_unit(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x1xf64>, %[[G:[^ ,:)]+]]: tensor<3x1xf64>) -> tensor<4x27x3xf64> {
// CHECK-NEXT: %[[BR:[^ ,:)]+]] = stablehlo.reshape %[[B]] : (tensor<3x1xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[GR:[^ ,:)]+]] = stablehlo.reshape %[[G]] : (tensor<3x1xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[BR]], dims = [1] : (tensor<3xf64>) -> tensor<5x3xf64>
// CHECK-NEXT: %[[G0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[GR]], dims = [1] : (tensor<3xf64>) -> tensor<22x3xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[B0]], %[[G0]], dim = 0 : (tensor<5x3xf64>, tensor<22x3xf64>) -> tensor<27x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[C]], dims = [1, 2] : (tensor<27x3xf64>) -> tensor<4x27x3xf64>
// CHECK-NEXT: return %[[R]] : tensor<4x27x3xf64>
// CHECK-NEXT: }
