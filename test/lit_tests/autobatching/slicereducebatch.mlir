// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %6 = stablehlo.broadcast_in_dim %0, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %7 = stablehlo.broadcast_in_dim %1, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %8 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %9 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %10 = stablehlo.broadcast_in_dim %4, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %11 = stablehlo.broadcast_in_dim %5, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %12 = stablehlo.reduce(%6 init: %cst) applies stablehlo.multiply across dimensions = [1, 2, 3] : (tensor<4x5x6x3xf32>, tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.reduce(%7 init: %cst) applies stablehlo.multiply across dimensions = [1, 2, 3] : (tensor<4x5x6x3xf32>, tensor<f32>) -> tensor<4xf32>
    %14 = stablehlo.reduce(%8 init: %cst) applies stablehlo.multiply across dimensions = [1, 2, 3] : (tensor<4x5x6x3xf32>, tensor<f32>) -> tensor<4xf32>
    %15 = stablehlo.reduce(%9 init: %cst) applies stablehlo.multiply across dimensions = [1, 2, 3] : (tensor<4x5x6x3xf32>, tensor<f32>) -> tensor<4xf32>
    %16 = stablehlo.reduce(%10 init: %cst) applies stablehlo.multiply across dimensions = [1, 2, 3] : (tensor<4x5x6x3xf32>, tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.reduce(%11 init: %cst) applies stablehlo.multiply across dimensions = [1, 2, 3] : (tensor<4x5x6x3xf32>, tensor<f32>) -> tensor<4xf32>
    return %12, %13, %14, %15, %16, %17 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<3.000000e+01> : tensor<6x4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<6x3xf32>) -> tensor<6x4x3xf32>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.multiply across dimensions = [2] : (tensor<6x4x3xf32>, tensor<f32>) -> tensor<6x4xf32>
// CHECK-NEXT:     %2 = stablehlo.power %1, %cst : tensor<6x4xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %4 = stablehlo.reshape %3 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %5 = stablehlo.slice %2 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %6 = stablehlo.reshape %5 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %7 = stablehlo.slice %2 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %9 = stablehlo.slice %2 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %10 = stablehlo.reshape %9 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %11 = stablehlo.slice %2 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %12 = stablehlo.reshape %11 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %13 = stablehlo.slice %2 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %14 = stablehlo.reshape %13 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     return %14, %12, %10, %8, %6, %4 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
