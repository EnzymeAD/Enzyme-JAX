// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzymexla-stablehlo-to-triton-compatible-dialect)" %s | FileCheck %s

func.func @main() -> tensor<8xi64> {
    %c_0 = stablehlo.constant dense<8> : tensor<8xi64>
    return %c_0 : tensor<8xi64>
}

// CHECK: func.func @main() -> tensor<8xi64> {
// CHECK-NEXT:     %cst = arith.constant dense<8> : tensor<8xi64>
// CHECK-NEXT:     return %cst : tensor<8xi64>
// CHECK-NEXT: }
