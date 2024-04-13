// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose(%928 : tensor<1x4x1x20x160xbf16>) -> tensor<1x1x48x160xbf16> {
  %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x48x4x20xbf16>
  %1097 = stablehlo.dot_general %cst_141, %928, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x48x4x20xbf16>, tensor<1x4x1x20x160xbf16>) -> tensor<1x1x48x160xbf16>
  return %1097 : tensor<1x1x48x160xbf16>
}

// CHECK:  func.func @transpose(%arg0: tensor<1x4x1x20x160xbf16>) -> tensor<1x1x48x160xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x48x160xbf16>
// CHECK-NEXT:    return %0 : tensor<1x1x48x160xbf16>
// CHECK-NEXT:  }
