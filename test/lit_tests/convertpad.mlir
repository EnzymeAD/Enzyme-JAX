// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @pad_multiply(%4: tensor<1x3x1024xf32>) -> tensor<1x3x2048xbf16> {
  %constant_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %5 = stablehlo.pad %4, %constant_0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %7 = stablehlo.convert %5 : (tensor<1x3x2048xf32>) -> tensor<1x3x2048xbf16>
  return %7 : tensor<1x3x2048xbf16>
}

// CHECK:  func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>) -> tensor<1x3x2048xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.convert %arg0 : (tensor<1x3x1024xf32>) -> tensor<1x3x1024xbf16>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xbf16>, tensor<bf16>) -> tensor<1x3x2048xbf16>
// CHECK-NEXT:    return %2 : tensor<1x3x2048xbf16>
// CHECK-NEXT:  }
