// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose() -> tensor<bf16> {
  %cst_188 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %725 = stablehlo.tanh %cst_188 : tensor<bf16>
  return %725 : tensor<bf16>
}

// CHECK:  func.func @transpose() -> tensor<bf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    return %[[i0]] : tensor<bf16>
// CHECK-NEXT:  }
