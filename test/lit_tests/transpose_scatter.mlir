// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-opt{passses=65536},enzyme-hlo-opt)' | FileCheck %s

func.func @main(%arg0: tensor<5x2xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4x3x2xf32>) -> tensor<5x2xf32> {
    %c = stablehlo.constant dense<[[[0, 1, 2, 3], [3, 1, 0, 2], [2, 4, 4, 2]]]> : tensor<1x3x4xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x2xf32>) -> tensor<2x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x3x2xf32>) -> tensor<2x3x4xf32>
    %2 = "stablehlo.scatter"(%0, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<2x5xf32>, tensor<1x3x4xi64>, tensor<2x3x4xf32>) -> tensor<2x5xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x5xf32>) -> tensor<5x2xf32>
    return %3 : tensor<5x2xf32>
}

// CHECK: func.func @main(%arg0: tensor<5x2xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4x3x2xf32>) -> tensor<5x2xf32> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[[0, 1, 2, 3], [3, 1, 0, 2], [2, 4, 4, 2]]]> : tensor<1x3x4xi64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x3x2xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:     %1 = "stablehlo.scatter"(%arg0, %c, %0) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:       %2 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<5x2xf32>, tensor<1x3x4xi64>, tensor<2x3x4xf32>) -> tensor<5x2xf32>
// CHECK-NEXT:     return %1 : tensor<5x2xf32>
// CHECK-NEXT: }
