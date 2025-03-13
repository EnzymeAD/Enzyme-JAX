// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main() -> tensor<3xi1> {
    %c = stablehlo.constant dense<[false, true, false]> : tensor<3xi1>
    %c1 = stablehlo.constant dense<[true, true, true]> : tensor<3xi1>
    %0 = stablehlo.and %c, %c1 : tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}

// CHECK:  func.func @main() -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[false, true, false]> : tensor<3xi1>
// CHECK-NEXT:    return %c : tensor<3xi1>
// CHECK-NEXT:  }
