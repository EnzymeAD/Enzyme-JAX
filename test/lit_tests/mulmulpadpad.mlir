// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @pad_multiply(%p1: tensor<1x3x1024xf32>,%p2: tensor<1x3x1024xf32>, %v1: tensor<f32>, %v2: tensor<f32>,  %2: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
  %pad1 = stablehlo.pad %p1, %v1, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %pad2 = stablehlo.pad %p2, %v2, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %mul1 = stablehlo.multiply %pad1, %2 : tensor<1x3x2048xf32>
  %mul2 = stablehlo.multiply %mul1, %pad2 : tensor<1x3x2048xf32>
  return %mul2 : tensor<1x3x2048xf32>
}

// CHECK:  func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<1x3x1024xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg3, %arg2 : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg1, %arg0 : tensor<1x3x1024xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// CHECK-NEXT:    %3 = stablehlo.multiply %arg4, %2 : tensor<1x3x2048xf32>
// CHECK-NEXT:    return %3 : tensor<1x3x2048xf32>
// CHECK-NEXT:  }
