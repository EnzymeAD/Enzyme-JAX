// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func public @main(%arg0: tensor<1x32xbf16>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reshape %arg0 : (tensor<1x32xbf16>) -> tensor<1x1x32xbf16>
    %3 = stablehlo.convert %2 : (tensor<1x1x32xbf16>) -> tensor<1x1x32xf32>
    %6 = stablehlo.reduce(%3 init: %1) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }
}

// CHECK:  func.func public @main(%arg0: tensor<1x32xbf16>) -> tensor<f32> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.convert %arg0 : (tensor<1x32xbf16>) -> tensor<1x32xf32>
// CHECK-NEXT:    %2 = stablehlo.reduce(%1 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %2 : tensor<f32>
// CHECK-NEXT:  }
