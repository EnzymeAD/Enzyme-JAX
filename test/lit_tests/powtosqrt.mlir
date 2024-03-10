// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2x2xf32>) -> tensor<2x2xf32> {
    %half = stablehlo.constant dense<5.000000e-01> : tensor<2x2xf32>
    %pd = stablehlo.power %a, %half : tensor<2x2xf32>
    return %pd : tensor<2x2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.sqrt %arg0 : tensor<2x2xf32>
// CHECK-NEXT:    return %0 : tensor<2x2xf32>
// CHECK-NEXT:  }
