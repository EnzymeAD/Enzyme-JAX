// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main() -> tensor<4xf64> {
  %c = stablehlo.constant dense<[0.0, 0.5, -0.5, 0.25]> : tensor<4xf64>
  %x = chlo.atanh %c : tensor<4xf64> -> tensor<4xf64>
  return %x : tensor<4xf64>
}

// CHECK:  func.func @main() -> tensor<4xf64> {
// CHECK-NEXT:    %[[c:.+]] = stablehlo.constant dense<[0.0, 0.5493061443340549, -0.5493061443340549, 0.2554128118829953]> : tensor<4xf64>
// CHECK-NEXT:    return %[[c]] : tensor<4xf64>
// CHECK-NEXT:  }
