// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @extend_operations(%arg28: tensor<1x24x96xf64>, %5290: tensor<1x8x80xf64>) -> (tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>) {
  %11825 = stablehlo.slice %5290 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
  %11826 = stablehlo.slice %5290 [0:1, 0:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
  %11827 = stablehlo.slice %arg28 [0:1, 17:24, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x7x80xf64>
  %11828 = "enzymexla.extend"(%5290) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x10x80xf64>
  %11829 = "enzymexla.extend"(%11826) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x8xf64>) -> tensor<1x10x8xf64>
  %11830 = "enzymexla.extend"(%11825) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x8xf64>) -> tensor<1x10x8xf64>

  return %11828, %11829, %11830 : tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>
}

// CHECK:     module {
// CHECK-NEXT:   func.func @extend_operations(%arg0: tensor<1x24x96xf64>, %arg1: tensor<1x8x80xf64>) -> (tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>) {
// CHECK-NEXT:     %0 = "enzymexla.extend"(%arg1) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:     %1 = "enzymexla.extend"(%arg1) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:1, 0:10, 0:8] : (tensor<1x10x80xf64>) -> tensor<1x10x8xf64>
// CHECK-NEXT:     %3 = "enzymexla.extend"(%arg1) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:     %4 = stablehlo.slice %3 [0:1, 0:10, 72:80] : (tensor<1x10x80xf64>) -> tensor<1x10x8xf64>
// CHECK-NEXT:     return %0, %2, %4 : tensor<1x10x80xf64>, tensor<1x10x8xf64>, tensor<1x10x8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
