// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=extend_splat --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main() -> tensor<14xf32> {
    %cst = stablehlo.constant dense<42.0> : tensor<10xf32>
    %0 = "enzymexla.extend"(%cst) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
    return %0 : tensor<14xf32>
  }
}

// CHECK:  func.func @main() -> tensor<14xf32> {
// CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<4.200000e+01> : tensor<14xf32>
// CHECK-NEXT:    return %[[CST]] : tensor<14xf32>
// CHECK-NEXT:  }
