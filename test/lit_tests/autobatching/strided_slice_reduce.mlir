// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<5x8x20x4xf32>) -> tensor<1x1x1x1xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x8x20x4xf32>) -> tensor<4x20x8x5xf32>
  %1 = stablehlo.slice %0 [0:3, 1:2, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
  %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<1xf32>
  %3 = stablehlo.slice %0 [0:3, 4:5, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
  %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<1xf32>
  %5 = stablehlo.slice %0 [0:3, 7:8, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
  %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<1xf32>
  %7 = stablehlo.slice %0 [0:3, 10:11, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
  %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<1xf32>
  %9 = stablehlo.slice %0 [0:3, 13:14, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
  %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<1xf32>
  %11 = stablehlo.add %2, %4 : tensor<1xf32>
  %12 = stablehlo.add %11, %6 : tensor<1xf32>
  %13 = stablehlo.add %12, %8 : tensor<1xf32>
  %14 = stablehlo.add %13, %10 : tensor<1xf32>
  %15 = stablehlo.reshape %14 : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
  return %15 : tensor<1x1x1x1xf32>
}

// CHECK:    func.func @main(%arg0: tensor<5x8x20x4xf32>) -> tensor<1x1x1x1xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x8x20x4xf32>) -> tensor<4x20x8x5xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:3, 1:2, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [0:3, 4:5, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [0:3, 7:8, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
// CHECK-NEXT:    %4 = stablehlo.slice %0 [0:3, 10:11, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
// CHECK-NEXT:    %5 = stablehlo.slice %0 [0:3, 13:14, 1:7, 1:2] : (tensor<4x20x8x5xf32>) -> tensor<3x1x6x1xf32>
// CHECK-NEXT:    %6 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3, 1] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %7 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3, 1] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %8 = stablehlo.add %6, %7 : tensor<f32>
// CHECK-NEXT:    %9 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3, 1] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<f32>
// CHECK-NEXT:    %11 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3, 1] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %12 = stablehlo.add %10, %11 : tensor<f32>
// CHECK-NEXT:    %13 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3, 1] : (tensor<3x1x6x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %14 = stablehlo.add %12, %13 : tensor<f32>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1x1x1x1xf32>
// CHECK-NEXT:    return %15 : tensor<1x1x1x1xf32>
// CHECK-NEXT:  }
