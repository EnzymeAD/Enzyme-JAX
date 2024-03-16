// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
  %0 = stablehlo.constant dense<3.600000e+00> : tensor<1x3x2048xf32>
  %1 = stablehlo.pad %arg0, %arg1, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<1x3x2048xf32>
  return %2 : tensor<1x3x2048xf32>
}

// CHECK:  func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<3.600000e+00> : tensor<1x3x1024xf32>
// CHECK-NEXT:    %1 = stablehlo.constant dense<3.600000e+00> : tensor<f32>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %arg1 : tensor<f32>
// CHECK-NEXT:    %3 = stablehlo.multiply %0, %arg0 : tensor<1x3x1024xf32>
// CHECK-NEXT:    %4 = stablehlo.pad %3, %2, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// CHECK-NEXT:    return %4 : tensor<1x3x2048xf32>
// CHECK-NEXT:  }
