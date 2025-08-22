// RUN: enzymexlamlir-opt --enzyme-batch --inline --enzyme-hlo-opt %s | FileCheck %s

module {
    func.func private @unbatched_reduce_window(%arg0: tensor<8x8xf32>) -> tensor<5x5xf32> {
        %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>, window_dimensions = array<i64: 2, 2>, window_strides = array<i64: 2, 2>}> ({
        ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1 : tensor<f32>
        }) : (tensor<8x8xf32>, tensor<f32>) -> tensor<5x5xf32>
        return %0 : tensor<5x5xf32>
    }
    func.func @main(%arg0: tensor<2x3x8x8xf32>) -> tensor<2x3x5x5xf32> {
        %0 = enzyme.batch @unbatched_reduce_window(%arg0) {batch_shape = array<i64: 2, 3>} : (tensor<2x3x8x8xf32>) -> tensor<2x3x5x5xf32>
        return %0 : tensor<2x3x5x5xf32>
    }
}

// CHECK: func.func @main(%arg0: tensor<2x3x8x8xf32>) -> tensor<2x3x5x5xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT{LITERAL}:     %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %1 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<2x3x8x8xf32>, tensor<f32>) -> tensor<2x3x5x5xf32>
// CHECK-NEXT:     return %0 : tensor<2x3x5x5xf32>
// CHECK-NEXT: }
