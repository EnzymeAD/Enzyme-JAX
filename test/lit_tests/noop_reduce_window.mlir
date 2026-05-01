// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Noop reduce_window: all window dims = 1, no padding, add with zero init.
// The result has the same shape as the input, so it should be replaced by the
// input.

func.func @noop_add_f64(%arg0: tensor<4x752x1520xf64>) -> tensor<4x752x1520xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
      base_dilations = array<i64: 1, 1, 1>,
      padding = dense<0> : tensor<3x2xi64>,
      window_dilations = array<i64: 1, 1, 1>,
      window_dimensions = array<i64: 1, 1, 1>,
      window_strides = array<i64: 1, 1, 1>}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %1 : tensor<f64>
  }) : (tensor<4x752x1520xf64>, tensor<f64>) -> tensor<4x752x1520xf64>
  return %0 : tensor<4x752x1520xf64>
}
// CHECK-LABEL: func.func @noop_add_f64
// CHECK-NEXT:   return %arg0 : tensor<4x752x1520xf64>

func.func @noop_add_f32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
      window_dimensions = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @noop_add_f32
// CHECK-NEXT:   return %arg0 : tensor<2x3xf32>

func.func @noop_max_f32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // -infinity for f32
  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
      window_dimensions = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @noop_max_f32
// CHECK-NEXT:   return %arg0 : tensor<2x3xf32>

func.func @noop_min_f32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // +infinity for f32
  %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
      window_dimensions = array<i64: 1, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.minimum %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @noop_min_f32
// CHECK-NEXT:   return %arg0 : tensor<2x3xf32>
