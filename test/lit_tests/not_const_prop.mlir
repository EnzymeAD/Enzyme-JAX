// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main() -> tensor<3xi1> {
    %c = stablehlo.constant dense<[false, true, false]> : tensor<3xi1>
    %0 = stablehlo.not %c : tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}

// CHECK:  func.func @main() -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
// CHECK-NEXT:    return %c : tensor<3xi1>
// CHECK-NEXT:  }
