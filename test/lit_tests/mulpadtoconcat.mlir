// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%1715 : tensor<1x8x3x1024x1024xbf16>, %1714: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x8x3x1024x2048xbf16> {
    %83 = stablehlo.constant dense<0.0> : tensor<bf16> 
    %1716 = stablehlo.pad %1715, %83, low = [0, 0, 0, 0, 1024], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x8x3x1024x1024xbf16>, tensor<bf16>) -> tensor<1x8x3x1024x2048xbf16>
    %1717 = stablehlo.add %1714, %1716 : tensor<1x8x3x1024x2048xbf16>
    return %1717 : tensor<1x8x3x1024x2048xbf16>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x8x3x1024x1024xbf16>, %arg1: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x8x3x1024x2048xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 0:8, 0:3, 0:1024, 0:1024] : (tensor<1x8x3x1024x2048xbf16>) -> tensor<1x8x3x1024x1024xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 0:8, 0:3, 0:1024, 1024:2048] : (tensor<1x8x3x1024x2048xbf16>) -> tensor<1x8x3x1024x1024xbf16>
// CHECK-NEXT:    %2 = stablehlo.add %1, %arg0 : tensor<1x8x3x1024x1024xbf16>
// CHECK-NEXT:    %3 = stablehlo.concatenate %0, %2, dim = 4 : (tensor<1x8x3x1024x1024xbf16>, tensor<1x8x3x1024x1024xbf16>) -> tensor<1x8x3x1024x2048xbf16>
// CHECK-NEXT:    return %3 : tensor<1x8x3x1024x2048xbf16>
// CHECK-NEXT:  }
