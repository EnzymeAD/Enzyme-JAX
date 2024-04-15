// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main() -> tensor<4xbf16> {
    %concat = stablehlo.constant dense<3.140000e+00> : tensor<4xf32>
    %conv = stablehlo.convert %concat : (tensor<4xf32>) -> tensor<4xbf16>
    return %conv : tensor<4xbf16>
  }
}

// CHECK:  func.func @main() -> tensor<4xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.140630e+00> : tensor<4xbf16>
// CHECK-NEXT:    return %[[i0]] : tensor<4xbf16>
// CHECK-NEXT:  }
