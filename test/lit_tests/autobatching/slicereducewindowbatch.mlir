// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

module @reactant_sliced_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<5x4x3xf32>) -> tensor<5x1x3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.slice %arg0 [0:5, 0:1, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %1 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %11 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<5x1x3xf32>, tensor<f32>) -> tensor<5x1x3xf32>
    %2 = stablehlo.slice %arg0 [0:5, 1:2, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %3 = "stablehlo.reduce_window"(%2, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %11 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<5x1x3xf32>, tensor<f32>) -> tensor<5x1x3xf32>
    %4 = stablehlo.slice %arg0 [0:5, 2:3, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %5 = "stablehlo.reduce_window"(%4, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %11 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<5x1x3xf32>, tensor<f32>) -> tensor<5x1x3xf32>
    %6 = stablehlo.slice %arg0 [0:5, 3:4, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x1x3xf32>
    %7 = "stablehlo.reduce_window"(%6, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [2, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %11 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<5x1x3xf32>, tensor<f32>) -> tensor<5x1x3xf32>
    %8 = stablehlo.add %1, %3 : tensor<5x1x3xf32>
    %9 = stablehlo.add %8, %5 : tensor<5x1x3xf32>
    %10 = stablehlo.add %9, %7 : tensor<5x1x3xf32>
    return %10 : tensor<5x1x3xf32>
  }
}

// CHECK: module @reactant_sliced_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:   func.func @main(%arg0: tensor<5x4x3xf32>) -> tensor<5x1x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0, 3] : (tensor<5x4x3xf32>) -> tensor<4x5x1x3xf32>
// CHECK-NEXT{LITERAL}:     %1 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [2, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 3>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %3 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<4x5x1x3xf32>, tensor<f32>) -> tensor<4x5x1x3xf32>
// CHECK-NEXT:     %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x5x1x3xf32>, tensor<f32>) -> tensor<5x1x3xf32>
// CHECK-NEXT:     return %2 : tensor<5x1x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
