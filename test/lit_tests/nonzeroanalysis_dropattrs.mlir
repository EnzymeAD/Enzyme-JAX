// RUN: enzymexlamlir-opt --drop-unsupported-attributes %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.exponential
// CHECK-NOT: enzymexla.non_zero
func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.exponential"(%x) {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
