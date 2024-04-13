// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose() -> tensor<1x1x20x4x48xbf16> {
  %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x48x4x20xbf16>
  %1072 = stablehlo.transpose %cst_141, dims = [0, 1, 4, 3, 2] : (tensor<1x1x48x4x20xbf16>) -> tensor<1x1x20x4x48xbf16> 
  return %1072 : tensor<1x1x20x4x48xbf16>
}

// CHECK:  func.func @transpose() -> tensor<1x1x20x4x48xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x20x4x48xbf16>
// CHECK-NEXT:    return %0 : tensor<1x1x20x4x48xbf16>
// CHECK-NEXT:  }
