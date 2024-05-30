// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @f() -> tensor<10xi32> {
  %c11 = stablehlo.constant dense<11> : tensor<10xi32>
  %c3 = stablehlo.constant dense<3> : tensor<10xi32>
  %7 = stablehlo.remainder %c11, %c3 : tensor<10xi32>
  return %7 : tensor<10xi32>
}

// CHECK:  func.func @f() -> tensor<10xi32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<10xi32>
// CHECK-NEXT:    return %c : tensor<10xi32>
// CHECK-NEXT:  }
