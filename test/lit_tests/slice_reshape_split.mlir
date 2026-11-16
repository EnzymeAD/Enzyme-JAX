// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_reshape_concat;slice_reshape_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A per-element scratch row, flattened, written as pieces that are not whole
// rows of 5, read back as rows of 5: rows 0 to 6 are the first 30 elements,
// the first piece.
func.func @rows_of_one_piece(%a: tensor<8x30xf64>, %b: tensor<8x6xf64>, %c: tensor<8x24xf64>) -> tensor<8x6x5xf64> {
  %cat = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<8x30xf64>, tensor<8x6xf64>, tensor<8x24xf64>) -> tensor<8x60xf64>
  %r = stablehlo.reshape %cat : (tensor<8x60xf64>) -> tensor<8x12x5xf64>
  %s = stablehlo.slice %r [0:8, 0:6, 0:5] : (tensor<8x12x5xf64>) -> tensor<8x6x5xf64>
  return %s : tensor<8x6x5xf64>
}

// CHECK-LABEL: func.func @rows_of_one_piece(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x30xf64>, %[[V1:[^ ,:)]+]]: tensor<8x6xf64>, %[[V2:[^ ,:)]+]]: tensor<8x24xf64>) -> tensor<8x6x5xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], dim = 1 : (tensor<8x30xf64>) -> tensor<8x30xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<8x30xf64>) -> tensor<8x6x5xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x6x5xf64>
// CHECK-NEXT: }

// Rows 3 and 4 are elements 15 to 24, all inside the padded operand.
func.func @rows_inside_pad_operand(%a: tensor<8x25xf64>) -> tensor<8x2x5xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 0], high = [0, 35], interior = [0, 0] : (tensor<8x25xf64>, tensor<f64>) -> tensor<8x60xf64>
  %r = stablehlo.reshape %p : (tensor<8x60xf64>) -> tensor<8x12x5xf64>
  %s = stablehlo.slice %r [0:8, 3:5, 0:5] : (tensor<8x12x5xf64>) -> tensor<8x2x5xf64>
  return %s : tensor<8x2x5xf64>
}

// CHECK-LABEL: func.func @rows_inside_pad_operand(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x25xf64>) -> tensor<8x2x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 15:25] : (tensor<8x25xf64>) -> tensor<8x10xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x10xf64>) -> tensor<8x2x5xf64>
// CHECK-NEXT: return %[[V2]] : tensor<8x2x5xf64>
// CHECK-NEXT: }

// Elements 1 to 3 of row 3 are elements 16 to 18.
func.func @part_of_one_row(%a: tensor<8x25xf64>) -> tensor<8x1x3xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 0], high = [0, 35], interior = [0, 0] : (tensor<8x25xf64>, tensor<f64>) -> tensor<8x60xf64>
  %r = stablehlo.reshape %p : (tensor<8x60xf64>) -> tensor<8x12x5xf64>
  %s = stablehlo.slice %r [0:8, 3:4, 1:4] : (tensor<8x12x5xf64>) -> tensor<8x1x3xf64>
  return %s : tensor<8x1x3xf64>
}

// CHECK-LABEL: func.func @part_of_one_row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x25xf64>) -> tensor<8x1x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 16:19] : (tensor<8x25xf64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x3xf64>) -> tensor<8x1x3xf64>
// CHECK-NEXT: return %[[V2]] : tensor<8x1x3xf64>
// CHECK-NEXT: }

// The first three elements of each row are not one run: nothing moves.
func.func @part_of_each_row(%a: tensor<8x30xf64>, %b: tensor<8x30xf64>) -> tensor<8x6x3xf64> {
  %cat = stablehlo.concatenate %a, %b, dim = 1 : (tensor<8x30xf64>, tensor<8x30xf64>) -> tensor<8x60xf64>
  %r = stablehlo.reshape %cat : (tensor<8x60xf64>) -> tensor<8x12x5xf64>
  %s = stablehlo.slice %r [0:8, 0:6, 0:3] : (tensor<8x12x5xf64>) -> tensor<8x6x3xf64>
  return %s : tensor<8x6x3xf64>
}

// CHECK-LABEL: func.func @part_of_each_row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x30xf64>, %[[V1:[^ ,:)]+]]: tensor<8x30xf64>) -> tensor<8x6x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 1 : (tensor<8x30xf64>, tensor<8x30xf64>) -> tensor<8x60xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x60xf64>) -> tensor<8x12x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V3]] [0:8, 0:6, 0:3] : (tensor<8x12x5xf64>) -> tensor<8x6x3xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x6x3xf64>
// CHECK-NEXT: }
