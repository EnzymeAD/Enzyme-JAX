// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=binop_pad_to_concat_add<1>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The add over the pad is read by slices that between them take every row:
// split, the pieces would only be put back together. Left as it is.
func.func @read_whole(%x: tensor<4x3xf64>, %y: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
  %c = stablehlo.constant dense<27.0> : tensor<f64>
  %p = stablehlo.pad %y, %c, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<4x3xf64>
  %a = stablehlo.add %x, %p : tensor<4x3xf64>
  %lo = stablehlo.slice %a [0:2, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
  %hi = stablehlo.slice %a [2:4, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
  return %lo, %hi : tensor<2x3xf64>, tensor<2x3xf64>
}

// CHECK:  func.func @read_whole(%arg0: tensor<4x3xf64>, %arg1: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.700000e+01> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %1 = stablehlo.add %arg0, %0 : tensor<4x3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:2, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %1 [2:4, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    return %2, %3 : tensor<2x3xf64>, tensor<2x3xf64>
// CHECK-NEXT:  }

// -----

// Only the padded rows are read: split, so that the slice takes its piece.
func.func @read_part(%x: tensor<4x3xf64>, %y: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %c = stablehlo.constant dense<27.0> : tensor<f64>
  %p = stablehlo.pad %y, %c, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<4x3xf64>
  %a = stablehlo.add %x, %p : tensor<4x3xf64>
  %hi = stablehlo.slice %a [2:4, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
  return %hi : tensor<2x3xf64>
}

// CHECK:  func.func @read_part(%arg0: tensor<4x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.700000e+01> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:2, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg1 : tensor<2x3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [2:4, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 : tensor<2x3xf64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %1, %4, dim = 0 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %5 [2:4, 0:3] : (tensor<4x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    return %6 : tensor<2x3xf64>
// CHECK-NEXT:  }
