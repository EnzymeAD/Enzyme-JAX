// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=short_row_reduce_unroll" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A row of 4: the sum of its elements.
func.func @row4(%x: tensor<8x3x4xf64>) -> tensor<8x3xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %r = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  return %r : tensor<8x3xf64>
}

// CHECK:  func.func @row4(%arg0: tensor<8x3x4xf64>) -> tensor<8x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:8, 0:3, 0:1] : (tensor<8x3x4xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<8x3x1xf64>) -> tensor<8x3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:8, 0:3, 1:2] : (tensor<8x3x4xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<8x3x1xf64>) -> tensor<8x3xf64>
// CHECK-NEXT:    %4 = stablehlo.add %1, %3 : tensor<8x3xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %arg0 [0:8, 0:3, 2:3] : (tensor<8x3x4xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<8x3x1xf64>) -> tensor<8x3xf64>
// CHECK-NEXT:    %7 = stablehlo.add %4, %6 : tensor<8x3xf64>
// CHECK-NEXT:    %8 = stablehlo.slice %arg0 [0:8, 0:3, 3:4] : (tensor<8x3x4xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<8x3x1xf64>) -> tensor<8x3xf64>
// CHECK-NEXT:    %10 = stablehlo.add %7, %9 : tensor<8x3xf64>
// CHECK-NEXT:    return %10 : tensor<8x3xf64>
// CHECK-NEXT:  }

// -----

// Two trailing dimensions of 2, and an init that is not the identity.
func.func @rows2x2(%x: tensor<8x2x2xf32>) -> tensor<8xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %r = stablehlo.reduce(%x init: %cst) applies stablehlo.maximum across dimensions = [1, 2] : (tensor<8x2x2xf32>, tensor<f32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// CHECK:  func.func @rows2x2(%arg0: tensor<8x2x2xf32>) -> tensor<8xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:8, 0:1, 0:1] : (tensor<8x2x2xf32>) -> tensor<8x1x1xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<8x1x1xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:8, 0:1, 1:2] : (tensor<8x2x2xf32>) -> tensor<8x1x1xf32>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<8x1x1xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %4 = stablehlo.maximum %1, %3 : tensor<8xf32>
// CHECK-NEXT:    %5 = stablehlo.slice %arg0 [0:8, 1:2, 0:1] : (tensor<8x2x2xf32>) -> tensor<8x1x1xf32>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<8x1x1xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %7 = stablehlo.maximum %4, %6 : tensor<8xf32>
// CHECK-NEXT:    %8 = stablehlo.slice %arg0 [0:8, 1:2, 1:2] : (tensor<8x2x2xf32>) -> tensor<8x1x1xf32>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<8x1x1xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %10 = stablehlo.maximum %7, %9 : tensor<8xf32>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
// CHECK-NEXT:    %12 = stablehlo.maximum %11, %10 : tensor<8xf32>
// CHECK-NEXT:    return %12 : tensor<8xf32>
// CHECK-NEXT:  }

// -----

// A leading dimension and a row of 3 are left to the reduce.
func.func @kept(%x: tensor<4x8xf64>, %y: tensor<8x3xf64>) -> (tensor<8xf64>, tensor<8xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %r = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x8xf64>, tensor<f64>) -> tensor<8xf64>
  %s = stablehlo.reduce(%y init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x3xf64>, tensor<f64>) -> tensor<8xf64>
  return %r, %s : tensor<8xf64>, tensor<8xf64>
}

// CHECK:  func.func @kept(%arg0: tensor<4x8xf64>, %arg1: tensor<8x3xf64>) -> (tensor<8xf64>, tensor<8xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x8xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:    %1 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x3xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:    return %0, %1 : tensor<8xf64>, tensor<8xf64>
// CHECK-NEXT:  }
