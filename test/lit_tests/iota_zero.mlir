// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main() -> tensor<3x2xi64> {
    %0 = stablehlo.iota dim = 2 : tensor<3x2x1xi64>
    %1 = stablehlo.reshape %0 : (tensor<3x2x1xi64>) -> tensor<3x2xi64>
    return %1 : tensor<3x2xi64>
  }
}

// CHECK:  func.func @main() -> tensor<3x2xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<3x2xi64>
// CHECK-NEXT:    return %c : tensor<3x2xi64>
// CHECK-NEXT:  }
