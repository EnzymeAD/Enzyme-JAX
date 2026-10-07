// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

// A tree reduction unrolled into halving steps: the reduce of the last step
// is the reduce of the whole input.
func.func @tree(%arg0: tensor<8xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dilations = array<i64: 4>, window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %r = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %r : tensor<f64>
  }) : (tensor<8xf64>, tensor<f64>) -> tensor<4xf64>
  %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 2>, window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %r = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %r : tensor<f64>
  }) : (tensor<4xf64>, tensor<f64>) -> tensor<2xf64>
  %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
  return %2 : tensor<f64>
}

// CHECK:  func.func @tree(%arg0: tensor<8xf64>) -> tensor<f64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   return %0 : tensor<f64>
// CHECK-NEXT: }

// -----

// Halving along one dimension of a 2-D input, reduced over both.
func.func @twodim(%arg0: tensor<3x8xf32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dilations = array<i64: 1, 4>, window_dimensions = array<i64: 1, 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %r = stablehlo.maximum %a, %b : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }) : (tensor<3x8xf32>, tensor<f32>) -> tensor<3x4xf32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<3x4xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK:  func.func @twodim(%arg0: tensor<3x8xf32>) -> tensor<f32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<3x8xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   return %0 : tensor<f32>
// CHECK-NEXT: }

// -----

// The first step pads the 31 real elements to 256 with the identity.
func.func @padded(%arg0: tensor<31xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 225]]> : tensor<1x2xi64>, window_dilations = array<i64: 128>, window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %r = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %r : tensor<f64>
  }) : (tensor<31xf64>, tensor<f64>) -> tensor<128xf64>
  %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 64>, window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %r = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %r : tensor<f64>
  }) : (tensor<128xf64>, tensor<f64>) -> tensor<64xf64>
  %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>
  return %2 : tensor<f64>
}

// CHECK:  func.func @padded(%arg0: tensor<31xf64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<31xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

// -----

// The reduce does not cover the halved dimension: the windows stay.
func.func @other_dim(%arg0: tensor<3x8xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dilations = array<i64: 1, 4>, window_dimensions = array<i64: 1, 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %r = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }) : (tensor<3x8xf32>, tensor<f32>) -> tensor<3x4xf32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK:  func.func @other_dim(%arg0: tensor<3x8xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dilations = array<i64: 1, 4>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<3x8xf32>, tensor<f32>) -> tensor<3x4xf32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:   return %1 : tensor<4xf32>
// CHECK-NEXT: }

// -----

// Not a halving: dilation 2 on a dimension of size 8 overlaps windows.
func.func @overlap(%arg0: tensor<8xf32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dilations = array<i64: 2>, window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %r = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }) : (tensor<8xf32>, tensor<f32>) -> tensor<6xf32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK:  func.func @overlap(%arg0: tensor<8xf32>) -> tensor<f32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dilations = array<i64: 2>, window_dimensions = array<i64: 2>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<8xf32>, tensor<f32>) -> tensor<6xf32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   return %1 : tensor<f32>
// CHECK-NEXT: }

// -----

// The reduce_window init is not the identity: not foldable.
func.func @nonidentity(%arg0: tensor<8xf32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %one) <{window_dilations = array<i64: 4>, window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %r = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }) : (tensor<8xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK:  func.func @nonidentity(%arg0: tensor<8xf32>) -> tensor<f32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{window_dilations = array<i64: 4>, window_dimensions = array<i64: 2>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<8xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   return %1 : tensor<f32>
// CHECK-NEXT: }
