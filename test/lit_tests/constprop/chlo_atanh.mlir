// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main() -> tensor<4xf64> {
  %c = stablehlo.constant dense<[0.0, 0.5, -0.5, 0.25]> : tensor<4xf64>
  %x = chlo.atanh %c : tensor<4xf64> -> tensor<4xf64>
  return %x : tensor<4xf64>
}

// CHECK-LABEL: @main()
// CHECK-NEXT: stablehlo.constant
// CHECK-NOT: chlo.atanh
