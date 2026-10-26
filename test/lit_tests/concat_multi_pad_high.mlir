// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_multipad},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

module {
  func.func @main(%a: tensor<1x2x4xf32>, %b: tensor<1x5x4xf32>) -> tensor<1x9x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pa = stablehlo.pad %a, %zero, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %pb = stablehlo.pad %b, %zero, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x7x4xf32>
    %c = stablehlo.concatenate %pa, %pb, dim = 1 : (tensor<1x2x4xf32>, tensor<1x7x4xf32>) -> tensor<1x9x4xf32>
    return %c : tensor<1x9x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x5x4xf32>) -> tensor<1x9x4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2x4xf32>, tensor<1x5x4xf32>) -> tensor<1x7x4xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x7x4xf32>, tensor<f32>) -> tensor<1x9x4xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 1 : (tensor<1x9x4xf32>) -> tensor<1x9x4xf32>
// CHECK-NEXT:    return %2 : tensor<1x9x4xf32>
// CHECK-NEXT:  }
