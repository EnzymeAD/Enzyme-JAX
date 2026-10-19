// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=transpose_reduce_window" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A swap carries the window, dilation and padding of each dimension along.
func.func @swap_with_padding(%arg0: tensor<4x8x12xf64>) -> tensor<12x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [1, 1], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 2, 1>, window_dimensions = array<i64: 1, 4, 1>, window_strides = array<i64: 1, 1, 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %r = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %r : tensor<f64>
  }) : (tensor<4x8x12xf64>, tensor<f64>) -> tensor<4x4x12xf64>
  %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<4x4x12xf64>) -> tensor<12x4x4xf64>
  return %1 : tensor<12x4x4xf64>
}

// CHECK:  func.func @swap_with_padding(%arg0: tensor<4x8x12xf64>) -> tensor<12x4x4xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<4x8x12xf64>) -> tensor<12x8x4xf64>
// CHECK-NEXT:   %1 = "stablehlo.reduce_window"(%0, %cst) <{padding = dense<{{\[\[}}0, 0], [1, 1], [0, 0{{\]\]}}> : tensor<3x2xi64>, window_dilations = array<i64: 1, 2, 1>, window_dimensions = array<i64: 1, 4, 1>, window_strides = array<i64: 1, 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12x8x4xf64>, tensor<f64>) -> tensor<12x4x4xf64>
// CHECK-NEXT:   return %1 : tensor<12x4x4xf64>
// CHECK-NEXT: }

// -----

// A cyclic permutation (not its own inverse): the window of operand
// dimension perm[i] must become the window of result dimension i.
func.func @rotate(%arg0: tensor<4x8x12xf64>) -> tensor<3x12x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dimensions = array<i64: 1, 6, 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %r = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %r : tensor<f64>
  }) : (tensor<4x8x12xf64>, tensor<f64>) -> tensor<4x3x12xf64>
  %1 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<4x3x12xf64>) -> tensor<3x12x4xf64>
  return %1 : tensor<3x12x4xf64>
}

// CHECK:  func.func @rotate(%arg0: tensor<4x8x12xf64>) -> tensor<3x12x4xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<4x8x12xf64>) -> tensor<8x12x4xf64>
// CHECK-NEXT:   %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dimensions = array<i64: 6, 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8x12x4xf64>, tensor<f64>) -> tensor<3x12x4xf64>
// CHECK-NEXT:   return %1 : tensor<3x12x4xf64>
// CHECK-NEXT: }
