// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main(%a : tensor<2x3x1x1xf32>) -> tensor<1x6x11xf32> {
    %pv = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [1, 2, 0, 0], high = [3, 4, 0, 0], interior = [0, 1, 0, 0] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<6x11x1x1xf32>
    %conv = stablehlo.reshape %pad : (tensor<6x11x1x1xf32>) -> tensor<1x6x11xf32>
    return %conv : tensor<1x6x11xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x3x1x1xf32>) -> tensor<1x6x11xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reshape %arg0 : (tensor<2x3x1x1xf32>) -> tensor<1x2x3xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 1, 2], high = [0, 3, 4], interior = [0, 0, 1] : (tensor<1x2x3xf32>, tensor<f32>) -> tensor<1x6x11xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<1x6x11xf32>
// CHECK-NEXT:  }
