// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_slice_stack" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Row 0 of %b, one element at a time, stacked along dimension 0: the row,
// transposed.
func.func @row(%b: tensor<3x4xf64>) -> tensor<4x1xf64> {
  %0 = stablehlo.slice %b [0:1, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %1 = stablehlo.slice %b [0:1, 1:2] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %2 = stablehlo.slice %b [0:1, 2:3] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %3 = stablehlo.slice %b [0:1, 3:4] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %r = stablehlo.concatenate %0, %1, %2, %3, dim = 0 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<4x1xf64>
  return %r : tensor<4x1xf64>
}

// CHECK-LABEL: func.func @row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>) -> tensor<4x1xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<3x4xf64>) -> tensor<3x4x1xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:1, 0:4, 0:1] : (tensor<3x4x1xf64>) -> tensor<1x4x1xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0, 2] : (tensor<1x4x1xf64>) -> tensor<4x1x1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<4x1x1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4x1xf64>
// CHECK-NEXT: }

// Windows 6:9, 15:18, 24:27 of dimension 1 (a 3-wide window every 9),
// concatenated along dimension 2: one slice of %x with dimension 1 split
// into 3 blocks of 9.
func.func @gapped(%x: tensor<2x27x3xf64>) -> tensor<2x3x9xf64> {
  %0 = stablehlo.slice %x [0:2, 6:9, 0:3] : (tensor<2x27x3xf64>) -> tensor<2x3x3xf64>
  %1 = stablehlo.slice %x [0:2, 15:18, 0:3] : (tensor<2x27x3xf64>) -> tensor<2x3x3xf64>
  %2 = stablehlo.slice %x [0:2, 24:27, 0:3] : (tensor<2x27x3xf64>) -> tensor<2x3x3xf64>
  %r = stablehlo.concatenate %0, %1, %2, dim = 2 : (tensor<2x3x3xf64>, tensor<2x3x3xf64>, tensor<2x3x3xf64>) -> tensor<2x3x9xf64>
  return %r : tensor<2x3x9xf64>
}

// CHECK-LABEL: func.func @gapped(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x27x3xf64>) -> tensor<2x3x9xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<2x27x3xf64>) -> tensor<2x3x9x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:2, 0:3, 6:9, 0:3] : (tensor<2x3x9x3xf64>) -> tensor<2x3x3x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [0, 2, 1, 3] : (tensor<2x3x3x3xf64>) -> tensor<2x3x3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<2x3x3x3xf64>) -> tensor<2x3x9xf64>
// CHECK-NEXT: return %[[V4]] : tensor<2x3x9xf64>
// CHECK-NEXT: }

// Adjacent along the concatenated dimension itself: left to concat_slice.
func.func @along_concat_dim(%x: tensor<8x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.slice %x [0:2, 0:4] : (tensor<8x4xf64>) -> tensor<2x4xf64>
  %1 = stablehlo.slice %x [2:4, 0:4] : (tensor<8x4xf64>) -> tensor<2x4xf64>
  %r = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<4x4xf64>
  return %r : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @along_concat_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x4xf64>) -> tensor<4x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:2, 0:4] : (tensor<8x4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [2:4, 0:4] : (tensor<8x4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V1]], %[[V2]], dim = 0 : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x4xf64>
// CHECK-NEXT: }

// The step (3) does not divide the extent (10): left alone.
func.func @uneven_blocks(%x: tensor<2x10xf64>) -> tensor<4x1xf64> {
  %0 = stablehlo.slice %x [0:2, 0:1] : (tensor<2x10xf64>) -> tensor<2x1xf64>
  %1 = stablehlo.slice %x [0:2, 3:4] : (tensor<2x10xf64>) -> tensor<2x1xf64>
  %r = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2x1xf64>, tensor<2x1xf64>) -> tensor<4x1xf64>
  return %r : tensor<4x1xf64>
}

// CHECK-LABEL: func.func @uneven_blocks(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x10xf64>) -> tensor<4x1xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:2, 0:1] : (tensor<2x10xf64>) -> tensor<2x1xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:2, 3:4] : (tensor<2x10xf64>) -> tensor<2x1xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V1]], %[[V2]], dim = 0 : (tensor<2x1xf64>, tensor<2x1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x1xf64>
// CHECK-NEXT: }

// The same reshape of three consecutive windows (each 9 rows, split into 3x3,
// with a unit dimension in front), stacked along that unit dimension: x itself,
// reshaped and transposed.
func.func @through_reshape(%x: tensor<4x27x3xf64>) -> tensor<3x4x3x3x3xf64> {
  %0 = stablehlo.slice %x [0:4, 0:9, 0:3] : (tensor<4x27x3xf64>) -> tensor<4x9x3xf64>
  %1 = stablehlo.slice %x [0:4, 9:18, 0:3] : (tensor<4x27x3xf64>) -> tensor<4x9x3xf64>
  %2 = stablehlo.slice %x [0:4, 18:27, 0:3] : (tensor<4x27x3xf64>) -> tensor<4x9x3xf64>
  %r0 = stablehlo.reshape %0 : (tensor<4x9x3xf64>) -> tensor<1x4x3x3x3xf64>
  %r1 = stablehlo.reshape %1 : (tensor<4x9x3xf64>) -> tensor<1x4x3x3x3xf64>
  %r2 = stablehlo.reshape %2 : (tensor<4x9x3xf64>) -> tensor<1x4x3x3x3xf64>
  %c = stablehlo.concatenate %r0, %r1, %r2, dim = 0 : (tensor<1x4x3x3x3xf64>, tensor<1x4x3x3x3xf64>, tensor<1x4x3x3x3xf64>) -> tensor<3x4x3x3x3xf64>
  return %c : tensor<3x4x3x3x3xf64>
}

// CHECK-LABEL: func.func @through_reshape(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x27x3xf64>) -> tensor<3x4x3x3x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4x27x3xf64>) -> tensor<4x3x9x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [1, 0, 2, 3] : (tensor<4x3x9x3xf64>) -> tensor<3x4x9x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<3x4x9x3xf64>) -> tensor<3x4x3x3x3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3x4x3x3x3xf64>
// CHECK-NEXT: }

// Two windows of a gapped row, each reshaped with a unit dimension in the
// middle, stacked along it.
func.func @through_reshape_middle(%x: tensor<5x12xf64>) -> tensor<5x2x2xf64> {
  %0 = stablehlo.slice %x [0:5, 1:3] : (tensor<5x12xf64>) -> tensor<5x2xf64>
  %1 = stablehlo.slice %x [0:5, 7:9] : (tensor<5x12xf64>) -> tensor<5x2xf64>
  %r0 = stablehlo.reshape %0 : (tensor<5x2xf64>) -> tensor<5x1x2xf64>
  %r1 = stablehlo.reshape %1 : (tensor<5x2xf64>) -> tensor<5x1x2xf64>
  %c = stablehlo.concatenate %r0, %r1, dim = 1 : (tensor<5x1x2xf64>, tensor<5x1x2xf64>) -> tensor<5x2x2xf64>
  return %c : tensor<5x2x2xf64>
}

// CHECK-LABEL: func.func @through_reshape_middle(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x12xf64>) -> tensor<5x2x2xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<5x12xf64>) -> tensor<5x2x6xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:5, 0:2, 1:3] : (tensor<5x2x6xf64>) -> tensor<5x2x2xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0, 2] : (tensor<5x2x2xf64>) -> tensor<2x5x2xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [1, 0, 2] : (tensor<2x5x2xf64>) -> tensor<5x2x2xf64>
// CHECK-NEXT: return %[[V4]] : tensor<5x2x2xf64>
// CHECK-NEXT: }

// Through a reshape and a broadcast that inserts the unit dimension, as BK2PARTIAL's
// input windows with dot formation on: the chain applied to the stacked windows.
func.func @through_views(%x: tensor<8x27x3xf64>) -> tensor<3x3x8x3x3xf64> {
  %s0 = stablehlo.slice %x [0:8, 0:9, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
  %s1 = stablehlo.slice %x [0:8, 9:18, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
  %s2 = stablehlo.slice %x [0:8, 18:27, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
  %r0 = stablehlo.reshape %s0 : (tensor<8x9x3xf64>) -> tensor<8x3x3x3xf64>
  %r1 = stablehlo.reshape %s1 : (tensor<8x9x3xf64>) -> tensor<8x3x3x3xf64>
  %r2 = stablehlo.reshape %s2 : (tensor<8x9x3xf64>) -> tensor<8x3x3x3xf64>
  %b0 = stablehlo.broadcast_in_dim %r0, dims = [2, 1, 3, 4] : (tensor<8x3x3x3xf64>) -> tensor<1x3x8x3x3xf64>
  %b1 = stablehlo.broadcast_in_dim %r1, dims = [2, 1, 3, 4] : (tensor<8x3x3x3xf64>) -> tensor<1x3x8x3x3xf64>
  %b2 = stablehlo.broadcast_in_dim %r2, dims = [2, 1, 3, 4] : (tensor<8x3x3x3xf64>) -> tensor<1x3x8x3x3xf64>
  %c = stablehlo.concatenate %b0, %b1, %b2, dim = 0 : (tensor<1x3x8x3x3xf64>, tensor<1x3x8x3x3xf64>, tensor<1x3x8x3x3xf64>) -> tensor<3x3x8x3x3xf64>
  return %c : tensor<3x3x8x3x3xf64>
}

// CHECK-LABEL: func.func @through_views(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x27x3xf64>) -> tensor<3x3x8x3x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x27x3xf64>) -> tensor<8x3x9x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [1, 0, 2, 3] : (tensor<8x3x9x3xf64>) -> tensor<3x8x9x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<3x8x9x3xf64>) -> tensor<3x8x3x3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [0, 3, 2, 4, 5] : (tensor<3x8x3x3x3xf64>) -> tensor<3x1x3x8x3x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<3x1x3x8x3x3xf64>) -> tensor<3x3x8x3x3xf64>
// CHECK-NEXT: return %[[V5]] : tensor<3x3x8x3x3xf64>
// CHECK-NEXT: }

// The operands view the slices differently: left alone.
func.func @different_views(%x: tensor<8x27x3xf64>) -> tensor<2x8x9x3xf64> {
  %s0 = stablehlo.slice %x [0:8, 0:9, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
  %s1 = stablehlo.slice %x [0:8, 9:18, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [1, 2, 3] : (tensor<8x9x3xf64>) -> tensor<1x8x9x3xf64>
  %r1 = stablehlo.reshape %s1 : (tensor<8x9x3xf64>) -> tensor<1x8x9x3xf64>
  %c = stablehlo.concatenate %b0, %r1, dim = 0 : (tensor<1x8x9x3xf64>, tensor<1x8x9x3xf64>) -> tensor<2x8x9x3xf64>
  return %c : tensor<2x8x9x3xf64>
}

// CHECK-LABEL: func.func @different_views(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x27x3xf64>) -> tensor<2x8x9x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 0:9, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 9:18, 0:3] : (tensor<8x27x3xf64>) -> tensor<8x9x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [1, 2, 3] : (tensor<8x9x3xf64>) -> tensor<1x8x9x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x9x3xf64>) -> tensor<1x8x9x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 0 : (tensor<1x8x9x3xf64>, tensor<1x8x9x3xf64>) -> tensor<2x8x9x3xf64>
// CHECK-NEXT: return %[[V5]] : tensor<2x8x9x3xf64>
// CHECK-NEXT: }

// Windows stacked at different times: a run of two windows read through the
// blocked reshape of the buffer ([N, 3, 25, 5], blocks 1 and 2, through a
// transpose), and a single window read through the merged reshape ([N, 75,
// 5], rows 0:25, then split into the block's shape and given a unit
// dimension). The single window is a whole block of the merged dimension,
// so it is block 0 of the blocked value; the run of three blocks is one
// slice of it (here the whole value) with the run's views.
func.func @straggler(%a: tensor<3472875xf64>) -> tensor<3x5x9261x5x5xf64> {
  %3 = stablehlo.reshape %a : (tensor<3472875xf64>) -> tensor<9261x75x5xf64>
  %7 = stablehlo.slice %3 [0:9261, 0:25, 0:5] : (tensor<9261x75x5xf64>) -> tensor<9261x25x5xf64>
  %8 = stablehlo.reshape %7 : (tensor<9261x25x5xf64>) -> tensor<9261x5x5x5xf64>
  %16 = stablehlo.broadcast_in_dim %8, dims = [2, 1, 3, 4] : (tensor<9261x5x5x5xf64>) -> tensor<1x5x9261x5x5xf64>
  %11 = stablehlo.reshape %a : (tensor<3472875xf64>) -> tensor<9261x3x25x5xf64>
  %12 = stablehlo.transpose %11, dims = [1, 0, 2, 3] : (tensor<9261x3x25x5xf64>) -> tensor<3x9261x25x5xf64>
  %13 = stablehlo.slice %12 [1:3, 0:9261, 0:25, 0:5] : (tensor<3x9261x25x5xf64>) -> tensor<2x9261x25x5xf64>
  %14 = stablehlo.reshape %13 : (tensor<2x9261x25x5xf64>) -> tensor<2x9261x5x5x5xf64>
  %15 = stablehlo.transpose %14, dims = [0, 2, 1, 3, 4] : (tensor<2x9261x5x5x5xf64>) -> tensor<2x5x9261x5x5xf64>
  %17 = stablehlo.concatenate %16, %15, dim = 0 : (tensor<1x5x9261x5x5xf64>, tensor<2x5x9261x5x5xf64>) -> tensor<3x5x9261x5x5xf64>
  return %17 : tensor<3x5x9261x5x5xf64>
}

// CHECK-LABEL: func.func @straggler(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<3472875xf64>) -> tensor<3x5x9261x5x5xf64> {
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<3472875xf64>) -> tensor<9261x3x25x5xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[B]], dims = [1, 0, 2, 3] : (tensor<9261x3x25x5xf64>) -> tensor<3x9261x25x5xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[T]] : (tensor<3x9261x25x5xf64>) -> tensor<3x9261x5x5x5xf64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [0, 2, 1, 3, 4] : (tensor<3x9261x5x5x5xf64>) -> tensor<3x5x9261x5x5xf64>
// CHECK-NEXT: return %[[U]] : tensor<3x5x9261x5x5xf64>
// CHECK-NEXT: }

// A run and a single window of one value read through the same views:
// concat_slice's (consecutive slices of one value), left alone here.
func.func @same_operand(%x: tensor<3x8xf64>) -> tensor<3x8xf64> {
  %0 = stablehlo.slice %x [0:1, 0:8] : (tensor<3x8xf64>) -> tensor<1x8xf64>
  %1 = stablehlo.reshape %0 : (tensor<1x8xf64>) -> tensor<1x8xf64>
  %2 = stablehlo.slice %x [1:3, 0:8] : (tensor<3x8xf64>) -> tensor<2x8xf64>
  %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<1x8xf64>, tensor<2x8xf64>) -> tensor<3x8xf64>
  return %3 : tensor<3x8xf64>
}

// CHECK-LABEL: func.func @same_operand(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<3x8xf64>) -> tensor<3x8xf64> {
// CHECK-NEXT: %[[S0:[^ ,:)]+]] = stablehlo.slice %[[X]] [0:1, 0:8] : (tensor<3x8xf64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reshape %[[S0]] : (tensor<1x8xf64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.slice %[[X]] [1:3, 0:8] : (tensor<3x8xf64>) -> tensor<2x8xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[R0]], %[[S2]], dim = 0 : (tensor<1x8xf64>, tensor<2x8xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: return %[[C]] : tensor<3x8xf64>
// CHECK-NEXT: }

// The single window read through a transposing broadcast above its slice
// (a unit dimension inserted at the concatenated one), the run through a
// transpose under the blocked reshape: both resolve to the same blocked
// value [8, 3, 3, 3, 3] of the buffer, read [1, 0, 2, 4, 3].
func.func @chains(%a: tensor<648xf64>) -> tensor<3x8x3x3x3xf64> {
  %0 = stablehlo.reshape %a : (tensor<648xf64>) -> tensor<8x9x3x3xf64>
  %1 = stablehlo.transpose %0, dims = [0, 1, 3, 2] : (tensor<8x9x3x3xf64>) -> tensor<8x9x3x3xf64>
  %2 = stablehlo.reshape %1 : (tensor<8x9x3x3xf64>) -> tensor<8x3x3x3x3xf64>
  %3 = stablehlo.transpose %2, dims = [1, 0, 2, 3, 4] : (tensor<8x3x3x3x3xf64>) -> tensor<3x8x3x3x3xf64>
  %4 = stablehlo.slice %3 [1:3, 0:8, 0:3, 0:3, 0:3] : (tensor<3x8x3x3x3xf64>) -> tensor<2x8x3x3x3xf64>
  %5 = stablehlo.broadcast_in_dim %0, dims = [1, 2, 4, 3] : (tensor<8x9x3x3xf64>) -> tensor<1x8x9x3x3xf64>
  %6 = stablehlo.slice %5 [0:1, 0:8, 0:3, 0:3, 0:3] : (tensor<1x8x9x3x3xf64>) -> tensor<1x8x3x3x3xf64>
  %7 = stablehlo.concatenate %6, %4, dim = 0 : (tensor<1x8x3x3x3xf64>, tensor<2x8x3x3x3xf64>) -> tensor<3x8x3x3x3xf64>
  return %7 : tensor<3x8x3x3x3xf64>
}

// CHECK-LABEL: func.func @chains(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<648xf64>) -> tensor<3x8x3x3x3xf64> {
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<648xf64>) -> tensor<8x9x3x3xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[B]], dims = [0, 1, 3, 2] : (tensor<8x9x3x3xf64>) -> tensor<8x9x3x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[T]] : (tensor<8x9x3x3xf64>) -> tensor<8x3x3x3x3xf64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 0, 2, 3, 4] : (tensor<8x3x3x3x3xf64>) -> tensor<3x8x3x3x3xf64>
// CHECK-NEXT: return %[[U]] : tensor<3x8x3x3x3xf64>
// CHECK-NEXT: }

// The single window a sub-range inside block 0 of the merged dimension
// (elements 3:6 of 81, a block being 27), the run the same sub-range of
// blocks 1 and 2: one slice of the blocked value.
func.func @sub_block(%a: tensor<5832xf64>) -> tensor<3x8x3x3x3xf64> {
  %0 = stablehlo.reshape %a : (tensor<5832xf64>) -> tensor<8x81x3x3xf64>
  %1 = stablehlo.transpose %0, dims = [0, 1, 3, 2] : (tensor<8x81x3x3xf64>) -> tensor<8x81x3x3xf64>
  %2 = stablehlo.reshape %1 : (tensor<8x81x3x3xf64>) -> tensor<8x3x27x3x3xf64>
  %3 = stablehlo.transpose %2, dims = [1, 0, 2, 3, 4] : (tensor<8x3x27x3x3xf64>) -> tensor<3x8x27x3x3xf64>
  %4 = stablehlo.slice %3 [1:3, 0:8, 3:6, 0:3, 0:3] : (tensor<3x8x27x3x3xf64>) -> tensor<2x8x3x3x3xf64>
  %5 = stablehlo.broadcast_in_dim %0, dims = [1, 2, 4, 3] : (tensor<8x81x3x3xf64>) -> tensor<1x8x81x3x3xf64>
  %6 = stablehlo.slice %5 [0:1, 0:8, 3:6, 0:3, 0:3] : (tensor<1x8x81x3x3xf64>) -> tensor<1x8x3x3x3xf64>
  %7 = stablehlo.concatenate %6, %4, dim = 0 : (tensor<1x8x3x3x3xf64>, tensor<2x8x3x3x3xf64>) -> tensor<3x8x3x3x3xf64>
  return %7 : tensor<3x8x3x3x3xf64>
}

// CHECK-LABEL: func.func @sub_block(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<5832xf64>) -> tensor<3x8x3x3x3xf64> {
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<5832xf64>) -> tensor<8x81x3x3xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[B]], dims = [0, 1, 3, 2] : (tensor<8x81x3x3xf64>) -> tensor<8x81x3x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[T]] : (tensor<8x81x3x3xf64>) -> tensor<8x3x27x3x3xf64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 0, 2, 3, 4] : (tensor<8x3x27x3x3xf64>) -> tensor<3x8x27x3x3xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[U]] [0:3, 0:8, 3:6, 0:3, 0:3] : (tensor<3x8x27x3x3xf64>) -> tensor<3x8x3x3x3xf64>
// CHECK-NEXT: return %[[S]] : tensor<3x8x3x3x3xf64>
// CHECK-NEXT: }
