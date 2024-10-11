// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @wrapped() -> (tensor<1x3xf32> {mhlo.layout_mode = "default"}) {
    %cst_0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.378200e+00], [0.000000e+00, 1.378200e+00, 0.000000e+00], [0.000000e+00, 1.378200e+00, 1.378200e+00], [1.378200e+00, 0.000000e+00, 0.000000e+00], [1.378200e+00, 0.000000e+00, 1.378200e+00], [1.378200e+00, 1.378200e+00, 0.000000e+00], [1.378200e+00, 1.378200e+00, 1.378200e+00]]> : tensor<8x3xf32>
    %10 = stablehlo.slice %cst_0 [1:2, 0:3] : (tensor<8x3xf32>) -> tensor<1x3xf32>
    return %10 : tensor<1x3xf32>
  }
}

// CHECK:  func.func @wrapped() -> (tensor<1x3xf32> {mhlo.layout_mode = "default"}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<{{\[\[}}0.000000e+00, 0.000000e+00, 1.378200e+00{{\]\]}}> : tensor<1x3xf32>
// CHECK-NEXT:    return %cst : tensor<1x3xf32>
// CHECK-NEXT:  }
