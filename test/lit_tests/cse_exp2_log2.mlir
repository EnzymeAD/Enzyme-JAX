// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_exp2<16>;cse_log2<16>" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// CHECK-LABEL: @cse_exp2
// CHECK: %[[E:.+]] = stablehlo.exp2 %arg0 : tensor<4xf32>
// CHECK-NOT: stablehlo.exp2
// CHECK: %[[R:.+]] = stablehlo.add %[[E]], %[[E]] : tensor<4xf32>
// CHECK: return %[[R]]
func.func @cse_exp2(%x : tensor<4xf32>) -> tensor<4xf32> {
  %a = stablehlo.exp2 %x : tensor<4xf32>
  %b = stablehlo.exp2 %x : tensor<4xf32>
  %r = stablehlo.add %a, %b : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: @cse_log2
// CHECK: %[[L:.+]] = stablehlo.log2 %arg0 : tensor<4xf32>
// CHECK-NOT: stablehlo.log2
// CHECK: %[[R:.+]] = stablehlo.add %[[L]], %[[L]] : tensor<4xf32>
// CHECK: return %[[R]]
func.func @cse_log2(%x : tensor<4xf32>) -> tensor<4xf32> {
  %a = stablehlo.log2 %x : tensor<4xf32>
  %b = stablehlo.log2 %x : tensor<4xf32>
  %r = stablehlo.add %a, %b : tensor<4xf32>
  return %r : tensor<4xf32>
}
