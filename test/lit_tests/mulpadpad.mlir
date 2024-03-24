// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @mulpadpad(%x: tensor<1x3x1024xf32>, %px: tensor<f32>, %y: tensor<1x3x1024xf32>, %py: tensor<f32>) -> tensor<1x3x2048xf32> {
  %xr = stablehlo.pad %x, %px, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %yr = stablehlo.pad %y, %py, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %7 = stablehlo.multiply %xr, %yr : tensor<1x3x2048xf32>
  return %7 : tensor<1x3x2048xf32>
}

// CHECK:  func.func @mulpadpad(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>, %arg2: tensor<1x3x1024xf32>, %arg3: tensor<f32>) -> tensor<1x3x2048xf32> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg1, %arg3 : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg0, %arg2 : tensor<1x3x1024xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// CHECK-NEXT:    return %2 : tensor<1x3x2048xf32>
// CHECK-NEXT:  }
