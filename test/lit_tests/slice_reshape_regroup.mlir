// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape;slice_reshape_concat;slice_reshape_pad" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// [12, 5] -> [2, 30] splits 12 into (2, 6) and merges (6, 5): row 1 is rows
// 6..11 of %x.
func.func @regroup(%x: tensor<12x5xf64>) -> tensor<1x30xf64> {
  %r = stablehlo.reshape %x : (tensor<12x5xf64>) -> tensor<2x30xf64>
  %s = stablehlo.slice %r [1:2, 0:30] : (tensor<2x30xf64>) -> tensor<1x30xf64>
  return %s : tensor<1x30xf64>
}

// CHECK-LABEL: func.func @regroup(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<12x5xf64>) -> tensor<1x30xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [6:12, 0:5] : (tensor<12x5xf64>) -> tensor<6x5xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<6x5xf64>) -> tensor<1x30xf64>
// CHECK-NEXT: return %[[V2]] : tensor<1x30xf64>
// CHECK-NEXT: }

// Part of a merged dim: columns 10..19 of the 30 are rows 2 and 3 of %x's
// inner (6, 5).
func.func @merged_rows(%x: tensor<4x6x5xf64>) -> tensor<4x10xf64> {
  %r = stablehlo.reshape %x : (tensor<4x6x5xf64>) -> tensor<4x30xf64>
  %s = stablehlo.slice %r [0:4, 10:20] : (tensor<4x30xf64>) -> tensor<4x10xf64>
  return %s : tensor<4x10xf64>
}

// CHECK-LABEL: func.func @merged_rows(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x6x5xf64>) -> tensor<4x10xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:4, 2:4, 0:5] : (tensor<4x6x5xf64>) -> tensor<4x2x5xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<4x2x5xf64>) -> tensor<4x10xf64>
// CHECK-NEXT: return %[[V2]] : tensor<4x10xf64>
// CHECK-NEXT: }

// Columns 3..7 cross a row of the inner (6, 5): not a slice of %x.
func.func @crosses_row(%x: tensor<4x6x5xf64>) -> tensor<4x5xf64> {
  %r = stablehlo.reshape %x : (tensor<4x6x5xf64>) -> tensor<4x30xf64>
  %s = stablehlo.slice %r [0:4, 3:8] : (tensor<4x30xf64>) -> tensor<4x5xf64>
  return %s : tensor<4x5xf64>
}

// CHECK-LABEL: func.func @crosses_row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x6x5xf64>) -> tensor<4x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4x6x5xf64>) -> tensor<4x30xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4, 3:8] : (tensor<4x30xf64>) -> tensor<4x5xf64>
// CHECK-NEXT: return %[[V2]] : tensor<4x5xf64>
// CHECK-NEXT: }

// A scratch assembled by a concatenate and read back one block at a time,
// through a regrouping reshape: each read is its piece, even with the other
// read still using the reshape.
func.func @concat_blocks(%a: tensor<6x5xf64>, %b: tensor<6x5xf64>) -> (tensor<1x30xf64>, tensor<1x30xf64>) {
  %c = stablehlo.concatenate %a, %b, dim = 0 : (tensor<6x5xf64>, tensor<6x5xf64>) -> tensor<12x5xf64>
  %r = stablehlo.reshape %c : (tensor<12x5xf64>) -> tensor<2x30xf64>
  %s0 = stablehlo.slice %r [0:1, 0:30] : (tensor<2x30xf64>) -> tensor<1x30xf64>
  %s1 = stablehlo.slice %r [1:2, 0:30] : (tensor<2x30xf64>) -> tensor<1x30xf64>
  return %s0, %s1 : tensor<1x30xf64>, tensor<1x30xf64>
}

// CHECK-LABEL: func.func @concat_blocks(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<6x5xf64>, %[[V1:[^ ,:)]+]]: tensor<6x5xf64>) -> (tensor<1x30xf64>, tensor<1x30xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<6x5xf64>, tensor<6x5xf64>) -> tensor<12x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:6, 0:5] : (tensor<12x5xf64>) -> tensor<6x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<6x5xf64>) -> tensor<1x30xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<6x5xf64>) -> tensor<1x30xf64>
// CHECK-NEXT: return %[[V4]], %[[V5]] : tensor<1x30xf64>, tensor<1x30xf64>
// CHECK-NEXT: }

// The same through a pad, with zero rows between the pieces (as ceed BP3's
// per-element scratch): each read is the first 5 rows of a 6-row block.
func.func @pad_blocks(%a: tensor<4x5x6xf64>, %b: tensor<4x5x6xf64>, %d: tensor<4x5x6xf64>) -> (tensor<4x1x30xf64>, tensor<4x1x30xf64>) {
  %z = stablehlo.constant dense<0.0> : tensor<4x1x6xf64>
  %zs = stablehlo.constant dense<0.0> : tensor<f64>
  %c = stablehlo.concatenate %a, %z, %b, %z, %d, dim = 1 : (tensor<4x5x6xf64>, tensor<4x1x6xf64>, tensor<4x5x6xf64>, tensor<4x1x6xf64>, tensor<4x5x6xf64>) -> tensor<4x17x6xf64>
  %p = stablehlo.pad %c, %zs, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x17x6xf64>, tensor<f64>) -> tensor<4x18x6xf64>
  %r = stablehlo.reshape %p : (tensor<4x18x6xf64>) -> tensor<4x3x36xf64>
  %s0 = stablehlo.slice %r [0:4, 0:1, 0:30] : (tensor<4x3x36xf64>) -> tensor<4x1x30xf64>
  %s1 = stablehlo.slice %r [0:4, 1:2, 0:30] : (tensor<4x3x36xf64>) -> tensor<4x1x30xf64>
  return %s0, %s1 : tensor<4x1x30xf64>, tensor<4x1x30xf64>
}

// CHECK-LABEL: func.func @pad_blocks(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x5x6xf64>, %[[V1:[^ ,:)]+]]: tensor<4x5x6xf64>, %[[V2:[^ ,:)]+]]: tensor<4x5x6xf64>) -> (tensor<4x1x30xf64>, tensor<4x1x30xf64>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<4x1x6xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V3]], %[[V1]], %[[V3]], %[[V2]], dim = 1 : (tensor<4x5x6xf64>, tensor<4x1x6xf64>, tensor<4x5x6xf64>, tensor<4x1x6xf64>, tensor<4x5x6xf64>) -> tensor<4x17x6xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.pad %[[V5]], %[[V4]], low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x17x6xf64>, tensor<f64>) -> tensor<4x18x6xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V6]] [0:4, 0:5, 0:6] : (tensor<4x18x6xf64>) -> tensor<4x5x6xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<4x5x6xf64>) -> tensor<4x1x30xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.slice %[[V5]] [0:4, 6:11, 0:6] : (tensor<4x17x6xf64>) -> tensor<4x5x6xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.reshape %[[V9]] : (tensor<4x5x6xf64>) -> tensor<4x1x30xf64>
// CHECK-NEXT: return %[[V8]], %[[V10]] : tensor<4x1x30xf64>, tensor<4x1x30xf64>
// CHECK-NEXT: }

// A slice taking parts of two pieces with the reshape used elsewhere: left
// alone (it would copy).
func.func @two_pieces_shared(%a: tensor<6x5xf64>, %b: tensor<6x5xf64>) -> (tensor<1x30xf64>, tensor<2x30xf64>) {
  %c = stablehlo.concatenate %a, %b, dim = 0 : (tensor<6x5xf64>, tensor<6x5xf64>) -> tensor<12x5xf64>
  %r = stablehlo.reshape %c : (tensor<12x5xf64>) -> tensor<4x15xf64>
  %s = stablehlo.slice %r [1:3, 0:15] : (tensor<4x15xf64>) -> tensor<2x15xf64>
  %t = stablehlo.reshape %s : (tensor<2x15xf64>) -> tensor<1x30xf64>
  %u = stablehlo.reshape %r : (tensor<4x15xf64>) -> tensor<2x30xf64>
  return %t, %u : tensor<1x30xf64>, tensor<2x30xf64>
}

// CHECK-LABEL: func.func @two_pieces_shared(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<6x5xf64>, %[[V1:[^ ,:)]+]]: tensor<6x5xf64>) -> (tensor<1x30xf64>, tensor<2x30xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<6x5xf64>, tensor<6x5xf64>) -> tensor<12x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<12x5xf64>) -> tensor<4x15xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V3]] [1:3, 0:15] : (tensor<4x15xf64>) -> tensor<2x15xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<2x15xf64>) -> tensor<1x30xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<4x15xf64>) -> tensor<2x30xf64>
// CHECK-NEXT: return %[[V5]], %[[V6]] : tensor<1x30xf64>, tensor<2x30xf64>
// CHECK-NEXT: }
