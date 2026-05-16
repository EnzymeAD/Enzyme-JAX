// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --apply-pdll-patterns="rewrite-extent=checked-global include-patterns=SliceExtendCommute" | FileCheck %s --check-prefix=PDLL-GLOBAL
// RUN: enzymexlamlir-opt --apply-pdll-patterns="rewrite-extent=pure-local include-patterns=SliceExtendCommute" | FileCheck %s --check-prefix=PDLL-LOCAL
// RUN: enzymexlamlir-opt --apply-pdll-patterns="rewrite-extent=checked-local include-patterns=SliceExtendCommute" | FileCheck %s --check-prefix=PDLL-LOCAL


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
// CHECK-NEXT:   %[[slice1:.*]] = stablehlo.slice %[[extend]] [0:1, 1:3, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x2x96xf64>
// CHECK-NEXT:   %[[slice2:.*]] = stablehlo.slice %[[extend]] [0:1, 0:10, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x10x96xf64>
// CHECK-NEXT:   return %[[slice1]], %[[slice2]], %[[extend]] : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
// CHECK-NEXT: }

// PDLL-GLOBAL:      func.func @extend_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
// PDLL-GLOBAL-NEXT:   %[[extend:.*]] = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// PDLL-GLOBAL-NEXT:   %[[slice1:.*]] = stablehlo.slice %[[extend:.*]] [0:1, 1:3, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x2x96xf64>
// PDLL-GLOBAL-NEXT:   %[[slice2:.*]] = stablehlo.slice %[[extend:.*]] [0:1, 0:10, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x10x96xf64>
// PDLL-GLOBAL-NEXT:   return %[[slice1:.*]], %[[slice2:.*]], %[[extend:.*]] : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
// PDLL-GLOBAL-NEXT: }

// PDLL-LOCAL:      func.func @extend_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>) {
// PDLL-LOCAL-NEXT:   %[[extend1:.*]] = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// PDLL-LOCAL-NEXT:   %[[extend2:.*]] = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// PDLL-LOCAL-NEXT:   %[[slice1:.*]] = stablehlo.slice %[[extend1:.*]] [0:1, 1:3, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x2x96xf64>
// PDLL-LOCAL-NEXT:   %[[slice2:.*]] = stablehlo.slice %[[extend2:.*]] [0:1, 0:10, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x10x96xf64>
// PDLL-LOCAL-NEXT:   %[[extend3:.*]] = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// PDLL-LOCAL-NEXT:   return %[[slice1:.*]], %[[slice2:.*]], %[[extend3:.*]] : tensor<1x2x96xf64>, tensor<1x10x96xf64>, tensor<1x10x96xf64>
// PDLL-LOCAL-NEXT: }
