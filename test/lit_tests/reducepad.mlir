// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main(%a : tensor<2x3x1xf32>, %b : tensor<f32>) -> tensor<6x1xf32> {
    %pv = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [1, 2, 0], high = [3, 4, 0], interior = [0, 1, 0] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<6x11x1xf32>
    %conv = stablehlo.reduce(%pad init: %b) applies stablehlo.add across dimensions = [1] : (tensor<6x11x1xf32>, tensor<f32>) -> tensor<6x1xf32>
    return %conv : tensor<6x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x3x1xf32>, %arg1: tensor<f32>) -> tensor<6x1xf32> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [1, 0], high = [3, 0], interior = [0, 0] : (tensor<2x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK-NEXT:    return %2 : tensor<6x1xf32>
// CHECK-NEXT:  }
