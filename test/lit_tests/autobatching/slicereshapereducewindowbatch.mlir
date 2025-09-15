// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

module @reactant_sliced_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<5x4x3xf32>) -> tensor<5x3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.slice %arg0 [0:5, 0:1, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %1 = stablehlo.reshape %0 : (tensor<5x1x3xf32>) -> tensor<5x3xf32>
    %2 = stablehlo.slice %arg0 [0:5, 2:3, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %3 = stablehlo.reshape %2 : (tensor<5x1x3xf32>) -> tensor<5x3xf32>
    %4 = stablehlo.slice %arg0 [0:5, 1:2, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %5 = stablehlo.reshape %4 : (tensor<5x1x3xf32>) -> tensor<5x3xf32>
    %6 = stablehlo.slice %arg0 [0:5, 3:4, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %7 = stablehlo.reshape %6 : (tensor<5x1x3xf32>) -> tensor<5x3xf32>
    %8 = "stablehlo.reduce_window"(%1, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [2, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 3>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %15 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %15 : tensor<f32>
    }) : (tensor<5x3xf32>, tensor<f32>) -> tensor<5x3xf32>
    %9 = "stablehlo.reduce_window"(%5, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [2, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 3>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %15 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %15 : tensor<f32>
    }) : (tensor<5x3xf32>, tensor<f32>) -> tensor<5x3xf32>
    %10 = "stablehlo.reduce_window"(%3, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [2, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 3>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %15 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %15 : tensor<f32>
    }) : (tensor<5x3xf32>, tensor<f32>) -> tensor<5x3xf32>
    %11 = "stablehlo.reduce_window"(%7, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [2, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 3>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %15 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %15 : tensor<f32>
    }) : (tensor<5x3xf32>, tensor<f32>) -> tensor<5x3xf32>
    %12 = stablehlo.add %8, %9 : tensor<5x3xf32>
    %13 = stablehlo.add %12, %10 : tensor<5x3xf32>
    %14 = stablehlo.add %13, %11 : tensor<5x3xf32>
    return %14 : tensor<5x3xf32>
  }
}

// CHECK: module @reactant_sliced_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:   func.func @main(%arg0: tensor<5x4x3xf32>) -> tensor<5x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<5x4x3xf32>) -> tensor<4x5x3xf32>
// CHECK-NEXT{LITERAL}:     %1 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %3 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<4x5x3xf32>, tensor<f32>) -> tensor<4x5x3xf32>
// CHECK-NEXT:     %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x5x3xf32>, tensor<f32>) -> tensor<5x3xf32>
// CHECK-NEXT:     return %2 : tensor<5x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
