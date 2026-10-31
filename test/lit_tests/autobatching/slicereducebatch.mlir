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

// CHECK:    func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+01> : tensor<6x4xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<6x3xf32>) -> tensor<6x1x3xf32>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.multiply across dimensions = [2] : (tensor<6x1x3xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:    %3 = stablehlo.power %2, %cst : tensor<6x4xf32>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %6 = stablehlo.slice %3 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %8 = stablehlo.slice %3 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %10 = stablehlo.slice %3 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %12 = stablehlo.slice %3 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %13 = stablehlo.reshape %12 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %14 = stablehlo.slice %3 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %15, %13, %11, %9, %7, %5 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT:  }
