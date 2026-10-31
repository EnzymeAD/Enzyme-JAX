// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main() -> tensor<4xf64> {
  %c = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.5]> : tensor<4xf64>
  %x = chlo.acosh %c : tensor<4xf64> -> tensor<4xf64>
  return %x : tensor<4xf64>
}

// CHECK-LABEL: @main()
// CHECK-NEXT: stablehlo.constant
// CHECK-NOT: chlo.acosh
