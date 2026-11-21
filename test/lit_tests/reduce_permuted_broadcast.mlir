// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_permuted_broadcast" --transform-interpreter --enzyme-hlo-remove-transform --canonicalize --split-input-file | FileCheck %s

// A weighted sum over the dimension a permuting broadcast moved to the front:
// computed in the layout of %a and %b, it needs no permutation at all.
func.func @chain(%a: tensor<8x2x3xf64>, %b: tensor<8x2x3xf64>, %w: tensor<2xf64>) -> tensor<8x2xf64> {
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %pa = stablehlo.broadcast_in_dim %a, dims = [1, 3, 0] : (tensor<8x2x3xf64>) -> tensor<3x8x1x2xf64>
  %pb = stablehlo.broadcast_in_dim %b, dims = [1, 3, 0] : (tensor<8x2x3xf64>) -> tensor<3x8x1x2xf64>
  %m = stablehlo.multiply %pa, %pb : tensor<3x8x1x2xf64>
  %s = stablehlo.slice %m [0:2, 0:8, 0:1, 0:2] : (tensor<3x8x1x2xf64>) -> tensor<2x8x1x2xf64>
  %wb = stablehlo.broadcast_in_dim %w, dims = [0] : (tensor<2xf64>) -> tensor<2x8x1x2xf64>
  %x = stablehlo.multiply %wb, %s : tensor<2x8x1x2xf64>
  %r = stablehlo.reduce(%x init: %c0) applies stablehlo.add across dimensions = [0, 2] : (tensor<2x8x1x2xf64>, tensor<f64>) -> tensor<8x2xf64>
  return %r : tensor<8x2xf64>
}

// CHECK:  func.func @chain(%arg0: tensor<8x2x3xf64>, %arg1: tensor<8x2x3xf64>, %arg2: tensor<2xf64>) -> tensor<8x2xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<2xf64>) -> tensor<8x2x2xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg0, %arg1 : tensor<8x2x3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:8, 0:2, 0:2] : (tensor<8x2x3xf64>) -> tensor<8x2x2xf64>
// CHECK-NEXT:    %3 = stablehlo.multiply %0, %2 : tensor<8x2x2xf64>
// CHECK-NEXT:    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x2x2xf64>, tensor<f64>) -> tensor<8x2xf64>
// CHECK-NEXT:    return %4 : tensor<8x2xf64>
// CHECK-NEXT:  }

// -----

// The dimensions that stay are permuted too: only the reduced result is
// transposed.
func.func @residual(%a: tensor<8x2x3xf64>) -> tensor<2x8xf64> {
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %one = stablehlo.constant dense<1.000000e+00> : tensor<3x2x1x8xf64>
  %pa = stablehlo.broadcast_in_dim %a, dims = [3, 1, 0] : (tensor<8x2x3xf64>) -> tensor<3x2x1x8xf64>
  %x = stablehlo.add %pa, %one : tensor<3x2x1x8xf64>
  %r = stablehlo.reduce(%x init: %c0) applies stablehlo.add across dimensions = [0, 2] : (tensor<3x2x1x8xf64>, tensor<f64>) -> tensor<2x8xf64>
  return %r : tensor<2x8xf64>
}

// CHECK:  func.func @residual(%arg0: tensor<8x2x3xf64>) -> tensor<2x8xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<8x2x3xf64>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %cst_0 : tensor<8x2x3xf64>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x2x3xf64>, tensor<f64>) -> tensor<8x2xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x2xf64>) -> tensor<2x8xf64>
// CHECK-NEXT:    return %2 : tensor<2x8xf64>
// CHECK-NEXT:  }

// -----

// The permuted value is also returned, so it is formed anyway.
func.func @kept(%a: tensor<8x2x3xf64>) -> (tensor<8x2xf64>, tensor<3x8x1x2xf64>) {
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %pa = stablehlo.broadcast_in_dim %a, dims = [1, 3, 0] : (tensor<8x2x3xf64>) -> tensor<3x8x1x2xf64>
  %n = stablehlo.negate %pa : tensor<3x8x1x2xf64>
  %r = stablehlo.reduce(%n init: %c0) applies stablehlo.add across dimensions = [0, 2] : (tensor<3x8x1x2xf64>, tensor<f64>) -> tensor<8x2xf64>
  return %r, %pa : tensor<8x2xf64>, tensor<3x8x1x2xf64>
}

// CHECK:  func.func @kept(%arg0: tensor<8x2x3xf64>) -> (tensor<8x2xf64>, tensor<3x8x1x2xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 3, 0] : (tensor<8x2x3xf64>) -> tensor<3x8x1x2xf64>
// CHECK-NEXT:    %1 = stablehlo.negate %0 : tensor<3x8x1x2xf64>
// CHECK-NEXT:    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 2] : (tensor<3x8x1x2xf64>, tensor<f64>) -> tensor<8x2xf64>
// CHECK-NEXT:    return %2, %0 : tensor<8x2xf64>, tensor<3x8x1x2xf64>
// CHECK-NEXT:  }
