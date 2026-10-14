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
