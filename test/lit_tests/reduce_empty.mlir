// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%34: tensor<2xf32>) -> tensor<2xf32> {
    %cst_1 = stablehlo.constant dense<0.000> : tensor<f32>
    %35 = stablehlo.reduce(%34 init: %cst_1) applies stablehlo.add across dimensions = [] : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
    return %35 : tensor<2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:    return %arg0 : tensor<2xf32>
// CHECK-NEXT:  }
