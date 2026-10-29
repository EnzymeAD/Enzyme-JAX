// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main() -> tensor<4xf64> {
  %c = stablehlo.constant dense<[0.0, 1.0, -1.0, 0.5]> : tensor<4xf64>
  %x = chlo.cosh %c : tensor<4xf64> -> tensor<4xf64>
  return %x : tensor<4xf64>
}

// CHECK:  func.func @main() -> tensor<4xf64> {
// CHECK-NEXT:    %[[c:.+]] = stablehlo.constant dense<[1.0, 1.5430806348152437, 1.5430806348152437, 1.1276259652063807]> : tensor<4xf64>
// CHECK-NEXT:    return %[[c]] : tensor<4xf64>
// CHECK-NEXT:  }
