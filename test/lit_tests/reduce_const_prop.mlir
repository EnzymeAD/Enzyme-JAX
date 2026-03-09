// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main() -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x3xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%cst init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %4 = stablehlo.reduce(%3 init: %cst_1) applies stablehlo.multiply across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    return %4 : tensor<f32>
  }
}

// CHECK:  func.func @main() -> tensor<f32> {
// CHECK-NEXT:    %[[cst:.+]] = stablehlo.constant dense<2.700000e+01> : tensor<f32>
// CHECK-NEXT:    return %[[cst]] : tensor<f32>
// CHECK-NEXT:  }
