// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @extend_operations(%arg0: tensor<1x24x96xf64>, %arg1: tensor<1x8x80xf64>) -> (tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>) {
  %0 = stablehlo.slice %arg1 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
  %1 = stablehlo.slice %arg1 [0:1, 0:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
  %2 = stablehlo.slice %arg0 [0:1, 17:24, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x7x80xf64>
  %3 = "enzymexla.extend"(%arg1) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x10x80xf64>
  %4 = "enzymexla.extend"(%1) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x8xf64>) -> tensor<1x10x8xf64>
  %5 = "enzymexla.extend"(%0) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x8xf64>) -> tensor<1x10x8xf64>
  return %3, %4, %5 : tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>
}

// CHECK:      func.func @extend_operations(%arg0: tensor<1x24x96xf64>, %arg1: tensor<1x8x80xf64>) -> (tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>) {
// CHECK-NEXT:   %0 = "enzymexla.extend"(%arg1) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:1, 0:10, 72:80] : (tensor<1x10x80xf64>) -> tensor<1x10x8xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %0 [0:1, 0:10, 0:8] : (tensor<1x10x80xf64>) -> tensor<1x10x8xf64>
// CHECK-NEXT:   return %0, %2, %1 : tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>
// CHECK-NEXT: }
