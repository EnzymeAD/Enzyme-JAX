// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_wrap" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @wrap_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
  %0 = stablehlo.slice %arg0 [0:1, 1:3, 0:80] : (tensor<1x10x80xf64>) -> tensor<1x2x80xf64>
  %1 = stablehlo.slice %arg0 [0:1, 0:10, 0:80] : (tensor<1x10x80xf64>) -> tensor<1x10x80xf64>
  %3 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x2x80xf64>) -> tensor<1x2x96xf64>
  %4 = "enzymexla.wrap"(%1) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
  %5 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
  return %3, %4, %5 : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
}

// CHECK:      func.func @wrap_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
// CHECK-NEXT:   %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:1, 0:10, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x10x96xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %0 [0:1, 1:3, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x2x96xf64>
// CHECK-NEXT:   return %2, %1, %0 : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
// CHECK-NEXT: }
