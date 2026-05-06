// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --apply-pdll-patterns="include-patterns=[SliceExtendCommute]" --cse | FileCheck %s --check-prefix=PDLL

func.func @extend_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
  %0 = stablehlo.slice %arg0 [0:1, 1:3, 0:80] : (tensor<1x10x80xf64>) -> tensor<1x2x80xf64>
  %1 = stablehlo.slice %arg0 [0:1, 0:10, 0:80] : (tensor<1x10x80xf64>) -> tensor<1x10x80xf64>
  %3 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x2x80xf64>) -> tensor<1x2x96xf64>
  %4 = "enzymexla.extend"(%1) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
  %5 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
  return %3, %4, %5 : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
}

// CHECK:      func.func @extend_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
// CHECK-NEXT:   %[[extend:.*]] = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// CHECK-NEXT:   %[[slice1:.*]] = stablehlo.slice %[[extend]] [0:1, 0:10, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x10x96xf64>
// CHECK-NEXT:   %[[slice2:.*]] = stablehlo.slice %[[extend]] [0:1, 1:3, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x2x96xf64>
// CHECK-NEXT:   return %[[slice2]], %[[slice1]], %[[extend]] : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
// CHECK-NEXT: }

// PDLL:      func.func @extend_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
// PDLL-NEXT:   %[[extend:.*]] = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// PDLL-NEXT:   %[[slice2:.*]] = stablehlo.slice %[[extend]] [0:1, 1:3, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x2x96xf64>
// PDLL-NEXT:   %[[slice1:.*]] = stablehlo.slice %[[extend]] [0:1, 0:10, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x10x96xf64>
// PDLL-NEXT:   return %[[slice2]], %[[slice1]], %[[extend]] : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
// PDLL-NEXT: }
