// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%11832: tensor<1x8x80xf64>, %6509 : tensor<1x8x80xf64>, %11833 : tensor<1x8x80xf64>) -> tensor<1x8x288xf64> {
      %11834 = "enzymexla.wrap"(%11832) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
      %11835 = "enzymexla.wrap"(%6509) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
      %11836 = "enzymexla.wrap"(%11833) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
      %11837 = stablehlo.concatenate %11834, %11835, %11836, dim = 2 : (tensor<1x8x96xf64>, tensor<1x8x96xf64>, tensor<1x8x96xf64>) -> tensor<1x8x288xf64>
      stablehlo.return %11837 : tensor<1x8x288xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x8x80xf64>, %arg1: tensor<1x8x80xf64>, %arg2: tensor<1x8x80xf64>) -> tensor<1x8x288xf64> {
// CHECK-NEXT:    %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%arg1) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    %2 = "enzymexla.wrap"(%arg2) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 : (tensor<1x8x96xf64>, tensor<1x8x96xf64>, tensor<1x8x96xf64>) -> tensor<1x8x288xf64>
// CHECK-NEXT:    stablehlo.return %3 : tensor<1x8x288xf64>
// CHECK-NEXT:  }