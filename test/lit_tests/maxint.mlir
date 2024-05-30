// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @f() -> tensor<10xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<10xi32>
  %c2 = stablehlo.constant dense<2> : tensor<10xi32>
  %7 = stablehlo.maximum %c0, %c2 : tensor<10xi32>
  return %7 : tensor<10xi32>
}

// CHECK:  func.func @f() -> tensor<10xi32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<10xi32>
// CHECK-NEXT:    return %c : tensor<10xi32>
// CHECK-NEXT:  }
