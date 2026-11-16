// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_pad<1>;slice_concat<1>" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A per-element scratch row written one piece at a time is a pad of the
// pieces written so far; a sum-factorization stage reads it along another
// axis, with a stride. Positions 3, 9, ..., 33 all lie in the first piece:
// the read is a strided slice of that piece.
func.func @row_read_within_piece(%a: tensor<8x1x36xf64>, %b: tensor<8x1x36xf64>) -> tensor<8x1x6xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %c = stablehlo.concatenate %a, %b, dim = 2 : (tensor<8x1x36xf64>, tensor<8x1x36xf64>) -> tensor<8x1x72xf64>
  %p = stablehlo.pad %c, %z, low = [0, 0, 0], high = [0, 0, 144], interior = [0, 0, 0] : (tensor<8x1x72xf64>, tensor<f64>) -> tensor<8x1x216xf64>
  %s = stablehlo.slice %p [0:8, 0:1, 3:39:6] : (tensor<8x1x216xf64>) -> tensor<8x1x6xf64>
  return %s : tensor<8x1x6xf64>
}

// CHECK-LABEL: func.func @row_read_within_piece(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x1x36xf64>, %[[V1:[^ ,:)]+]]: tensor<8x1x36xf64>) -> tensor<8x1x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 0:1, 3:34:6] : (tensor<8x1x36xf64>) -> tensor<8x1x6xf64>
// CHECK-NEXT: return %[[V2]] : tensor<8x1x6xf64>
// CHECK-NEXT: }

// Positions 1, 9, ..., 89: the first before the padded operand, the last two
// past it, the rest in both pieces.
func.func @across_pad_and_pieces(%a: tensor<8x1x36xf64>, %b: tensor<8x1x36xf64>) -> tensor<8x1x12xf64> {
  %z = stablehlo.constant dense<1.5> : tensor<f64>
  %c = stablehlo.concatenate %a, %b, dim = 2 : (tensor<8x1x36xf64>, tensor<8x1x36xf64>) -> tensor<8x1x72xf64>
  %p = stablehlo.pad %c, %z, low = [0, 0, 4], high = [0, 0, 140], interior = [0, 0, 0] : (tensor<8x1x72xf64>, tensor<f64>) -> tensor<8x1x216xf64>
  %s = stablehlo.slice %p [0:8, 0:1, 1:91:8] : (tensor<8x1x216xf64>) -> tensor<8x1x12xf64>
  return %s : tensor<8x1x12xf64>
}

// CHECK-LABEL: func.func @across_pad_and_pieces(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x1x36xf64>, %[[V1:[^ ,:)]+]]: tensor<8x1x36xf64>) -> tensor<8x1x12xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<1.500000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 0:1, 5:36:8] : (tensor<8x1x36xf64>) -> tensor<8x1x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:8, 0:1, 1:34:8] : (tensor<8x1x36xf64>) -> tensor<8x1x5xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 2 : (tensor<8x1x4xf64>, tensor<8x1x5xf64>) -> tensor<8x1x9xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.pad %[[V5]], %[[V2]], low = [0, 0, 1], high = [0, 0, 2], interior = [0, 0, 0] : (tensor<8x1x9xf64>, tensor<f64>) -> tensor<8x1x12xf64>
// CHECK-NEXT: return %[[V6]] : tensor<8x1x12xf64>
// CHECK-NEXT: }

// Every position the stride reaches falls in the padding: the padding value.
func.func @only_padding(%a: tensor<8x10xf64>) -> tensor<8x3xf64> {
  %z = stablehlo.constant dense<2.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 0], high = [0, 20], interior = [0, 0] : (tensor<8x10xf64>, tensor<f64>) -> tensor<8x30xf64>
  %s = stablehlo.slice %p [0:8, 12:30:7] : (tensor<8x30xf64>) -> tensor<8x3xf64>
  return %s : tensor<8x3xf64>
}

// CHECK-LABEL: func.func @only_padding(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x10xf64>) -> tensor<8x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [] : (tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: return %[[V2]] : tensor<8x3xf64>
// CHECK-NEXT: }
