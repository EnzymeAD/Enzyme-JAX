// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @pad_multiply(%4: tensor<1x3x1024xf32>, %pv : tensor<f32>) -> tensor<1x3x2048xf32> {
  %5 = stablehlo.pad %4, %pv, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %7 = stablehlo.exponential %5 : (tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32>
  return %7 : tensor<1x3x2048xf32>
}

// CHECK:  func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
// CHECK-NEXT:    %0 = stablehlo.exponential %arg1 : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.exponential %arg0 : tensor<1x3x1024xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// CHECK-NEXT:    return %2 : tensor<1x3x2048xf32>
// CHECK-NEXT:  }
