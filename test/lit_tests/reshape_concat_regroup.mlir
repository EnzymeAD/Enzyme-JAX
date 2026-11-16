// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_concat" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Three 3-row pieces of a [9, 3, 3] dimension regrouped into 3 rows of 27:
// each piece is one row.
func.func @rows(%a: tensor<2x3x3x3xf64>, %b: tensor<2x3x3x3xf64>, %c: tensor<2x3x3x3xf64>) -> tensor<2x3x27xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<2x3x3x3xf64>, tensor<2x3x3x3xf64>, tensor<2x3x3x3xf64>) -> tensor<2x9x3x3xf64>
  %1 = stablehlo.reshape %0 : (tensor<2x9x3x3xf64>) -> tensor<2x3x27xf64>
  return %1 : tensor<2x3x27xf64>
}

// CHECK-LABEL: func.func @rows(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x3x3x3xf64>, %[[V1:[^ ,:)]+]]: tensor<2x3x3x3xf64>, %[[V2:[^ ,:)]+]]: tensor<2x3x3x3xf64>) -> tensor<2x3x27xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<2x3x3x3xf64>) -> tensor<2x1x27xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<2x3x3x3xf64>) -> tensor<2x1x27xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<2x3x3x3xf64>) -> tensor<2x1x27xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], %[[V5]], dim = 1 : (tensor<2x1x27xf64>, tensor<2x1x27xf64>, tensor<2x1x27xf64>) -> tensor<2x3x27xf64>
// CHECK-NEXT: return %[[V6]] : tensor<2x3x27xf64>
// CHECK-NEXT: }

// Uneven pieces that are still whole rows of 10: two rows and one row.
func.func @uneven(%a: tensor<4x5xf64>, %b: tensor<2x5xf64>) -> tensor<3x10xf64> {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x5xf64>, tensor<2x5xf64>) -> tensor<6x5xf64>
  %1 = stablehlo.reshape %0 : (tensor<6x5xf64>) -> tensor<3x10xf64>
  return %1 : tensor<3x10xf64>
}

// CHECK-LABEL: func.func @uneven(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x5xf64>, %[[V1:[^ ,:)]+]]: tensor<2x5xf64>) -> tensor<3x10xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4x5xf64>) -> tensor<2x10xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<2x5xf64>) -> tensor<1x10xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 0 : (tensor<2x10xf64>, tensor<1x10xf64>) -> tensor<3x10xf64>
// CHECK-NEXT: return %[[V4]] : tensor<3x10xf64>
// CHECK-NEXT: }

// A piece that is not a whole number of rows: left alone.
func.func @partial_row(%a: tensor<3x5xf64>, %b: tensor<3x5xf64>) -> tensor<3x10xf64> {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<3x5xf64>, tensor<3x5xf64>) -> tensor<6x5xf64>
  %1 = stablehlo.reshape %0 : (tensor<6x5xf64>) -> tensor<3x10xf64>
  return %1 : tensor<3x10xf64>
}

// CHECK-LABEL: func.func @partial_row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x5xf64>, %[[V1:[^ ,:)]+]]: tensor<3x5xf64>) -> tensor<3x10xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 0 : (tensor<3x5xf64>, tensor<3x5xf64>) -> tensor<6x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<6x5xf64>) -> tensor<3x10xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3x10xf64>
// CHECK-NEXT: }

// The concatenated dimension is not the outermost of its run: pieces would
// interleave, left alone.
func.func @inner_concat(%a: tensor<2x3x5xf64>, %b: tensor<2x3x5xf64>) -> tensor<12x5xf64> {
  %0 = stablehlo.concatenate %a, %b, dim = 1 : (tensor<2x3x5xf64>, tensor<2x3x5xf64>) -> tensor<2x6x5xf64>
  %1 = stablehlo.reshape %0 : (tensor<2x6x5xf64>) -> tensor<12x5xf64>
  return %1 : tensor<12x5xf64>
}

// CHECK-LABEL: func.func @inner_concat(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x3x5xf64>, %[[V1:[^ ,:)]+]]: tensor<2x3x5xf64>) -> tensor<12x5xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 1 : (tensor<2x3x5xf64>, tensor<2x3x5xf64>) -> tensor<2x6x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<2x6x5xf64>) -> tensor<12x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<12x5xf64>
// CHECK-NEXT: }

// Rows written as a few pieces (a 5-row block and a zero row, three times, as
// a per-element scratch): each 6-row group is one row of the result.
func.func @groups(%a: tensor<5x6xf64>, %b: tensor<5x6xf64>, %d: tensor<6x6xf64>, %z: tensor<1x6xf64>) -> tensor<3x36xf64> {
  %0 = stablehlo.concatenate %a, %z, %b, %z, %d, dim = 0 : (tensor<5x6xf64>, tensor<1x6xf64>, tensor<5x6xf64>, tensor<1x6xf64>, tensor<6x6xf64>) -> tensor<18x6xf64>
  %1 = stablehlo.reshape %0 : (tensor<18x6xf64>) -> tensor<3x36xf64>
  return %1 : tensor<3x36xf64>
}

// CHECK-LABEL: func.func @groups(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x6xf64>, %[[V1:[^ ,:)]+]]: tensor<5x6xf64>, %[[V2:[^ ,:)]+]]: tensor<6x6xf64>, %[[V3:[^ ,:)]+]]: tensor<1x6xf64>) -> tensor<3x36xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<5x6xf64>) -> tensor<1x30xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V4]], %[[V3]], dim = 1 : (tensor<1x30xf64>, tensor<1x6xf64>) -> tensor<1x36xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<5x6xf64>) -> tensor<1x30xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.concatenate %[[V6]], %[[V3]], dim = 1 : (tensor<1x30xf64>, tensor<1x6xf64>) -> tensor<1x36xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<6x6xf64>) -> tensor<1x36xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.concatenate %[[V5]], %[[V7]], %[[V8]], dim = 0 : (tensor<1x36xf64>, tensor<1x36xf64>, tensor<1x36xf64>) -> tensor<3x36xf64>
// CHECK-NEXT: return %[[V9]] : tensor<3x36xf64>
// CHECK-NEXT: }
