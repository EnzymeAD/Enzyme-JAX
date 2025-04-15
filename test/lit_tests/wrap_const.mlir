// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module @"reactant_loop!" {
  func.func @main() -> tensor<1x8x96xf64> {
    %cst_217 = stablehlo.constant dense<3.000000e+00> : tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%cst_217) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    stablehlo.return %1 : tensor<1x8x96xf64>
  }
}

// CHECK:    func.func @main() -> tensor<1x8x96xf64> {
// CHECK-NEXT:      %cst = stablehlo.constant dense<3.000000e+00> : tensor<1x8x96xf64>
// CHECK-NEXT:      stablehlo.return %cst : tensor<1x8x96xf64>
// CHECK-NEXT:    }