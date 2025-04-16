// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%11832: tensor<1x8x80xf64>, %6509 : tensor<1x8x80xf64>, %11833 : tensor<1x8x80xf64>) -> tensor<1x24x96xf64> {
      %11834 = "enzymexla.wrap"(%11832) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
      %11835 = "enzymexla.wrap"(%6509) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
      %11836 = "enzymexla.wrap"(%11833) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
      %11837 = stablehlo.concatenate %11834, %11835, %11836, dim = 1 : (tensor<1x8x96xf64>, tensor<1x8x96xf64>, tensor<1x8x96xf64>) -> tensor<1x24x96xf64>
      stablehlo.return %11837 : tensor<1x24x96xf64>
  }
}

// CHECK:    func.func @main(%arg0: tensor<1x8x80xf64>, %arg1: tensor<1x8x80xf64>, %arg2: tensor<1x8x80xf64>) -> tensor<1x24x96xf64> {
// CHECK-NEXT:      %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 1 : (tensor<1x8x80xf64>, tensor<1x8x80xf64>, tensor<1x8x80xf64>) -> tensor<1x24x80xf64>
// CHECK-NEXT:      %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x24x80xf64>) -> tensor<1x24x96xf64>
// CHECK-NEXT:      stablehlo.return %1 : tensor<1x24x96xf64>
// CHECK-NEXT:    }