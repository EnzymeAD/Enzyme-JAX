// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape_concat_window" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A 36-element piece written at offset 28 of a 156-element row (ceed BP3/2's
// per-element scratch), read back as a 3x4 window of the 13x12 view: rows 2-4,
// columns 4-7 are elements 0-3, 12-15 and 24-27 of the piece.
func.func @window(%a: tensor<8x28xf64>, %b: tensor<8x36xf64>, %c: tensor<8x92xf64>) -> tensor<8x3x4xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<8x28xf64>, tensor<8x36xf64>, tensor<8x92xf64>) -> tensor<8x156xf64>
  %1 = stablehlo.reshape %0 : (tensor<8x156xf64>) -> tensor<8x13x12xf64>
  %2 = stablehlo.slice %1 [0:8, 2:5, 4:8] : (tensor<8x13x12xf64>) -> tensor<8x3x4xf64>
  return %2 : tensor<8x3x4xf64>
}

// CHECK-LABEL: func.func @window(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x28xf64>, %[[V1:[^ ,:)]+]]: tensor<8x36xf64>, %[[V2:[^ ,:)]+]]: tensor<8x92xf64>) -> tensor<8x3x4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.pad %[[V1]], %[[V3]], low = [0, 4], high = [0, 8], interior = [0, 0] : (tensor<8x36xf64>, tensor<f64>) -> tensor<8x48xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<8x48xf64>) -> tensor<8x4x12xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.slice %[[V5]] [0:8, 0:3, 4:8] : (tensor<8x4x12xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: return %[[V6]] : tensor<8x3x4xf64>
// CHECK-NEXT: }

// A window that crosses into the next piece: left alone.
func.func @crosses(%a: tensor<8x28xf64>, %b: tensor<8x36xf64>, %c: tensor<8x92xf64>) -> tensor<8x3x4xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<8x28xf64>, tensor<8x36xf64>, tensor<8x92xf64>) -> tensor<8x156xf64>
  %1 = stablehlo.reshape %0 : (tensor<8x156xf64>) -> tensor<8x13x12xf64>
  %2 = stablehlo.slice %1 [0:8, 4:7, 4:8] : (tensor<8x13x12xf64>) -> tensor<8x3x4xf64>
  return %2 : tensor<8x3x4xf64>
}

// CHECK-LABEL: func.func @crosses(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x28xf64>, %[[V1:[^ ,:)]+]]: tensor<8x36xf64>, %[[V2:[^ ,:)]+]]: tensor<8x92xf64>) -> tensor<8x3x4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], %[[V2]], dim = 1 : (tensor<8x28xf64>, tensor<8x36xf64>, tensor<8x92xf64>) -> tensor<8x156xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<8x156xf64>) -> tensor<8x13x12xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V4]] [0:8, 4:7, 4:8] : (tensor<8x13x12xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x3x4xf64>
// CHECK-NEXT: }

// A piece that starts on a row and fills whole rows: no padding.
func.func @aligned(%a: tensor<8x24xf64>, %b: tensor<8x36xf64>) -> tensor<8x2x12xf64> {
  %0 = stablehlo.concatenate %a, %b, dim = 1 : (tensor<8x24xf64>, tensor<8x36xf64>) -> tensor<8x60xf64>
  %1 = stablehlo.reshape %0 : (tensor<8x60xf64>) -> tensor<8x5x12xf64>
  %2 = stablehlo.slice %1 [0:8, 3:5, 0:12] : (tensor<8x5x12xf64>) -> tensor<8x2x12xf64>
  return %2 : tensor<8x2x12xf64>
}

// CHECK-LABEL: func.func @aligned(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x24xf64>, %[[V1:[^ ,:)]+]]: tensor<8x36xf64>) -> tensor<8x2x12xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x36xf64>) -> tensor<8x3x12xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:8, 1:3, 0:12] : (tensor<8x3x12xf64>) -> tensor<8x2x12xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x2x12xf64>
// CHECK-NEXT: }
