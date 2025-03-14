// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main() -> tensor<3xi1> {
    %c = stablehlo.constant dense<[0x7f800000, 0x7fc00000, 1.0]> : tensor<3xf32>
    %0 = stablehlo.is_finite %c : (tensor<3xf32>) -> tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}

// CHECK:  func.func @main() -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[false, false, true]> : tensor<3xi1>
// CHECK-NEXT:    return %c : tensor<3xi1>
// CHECK-NEXT:  }
