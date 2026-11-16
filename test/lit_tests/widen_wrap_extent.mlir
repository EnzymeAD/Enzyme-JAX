// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=widen_wrap" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The wrap takes its `lhs` elements from the end of the operand: the slice
// next to a wrap that already takes all of them has nothing left to fold in.
func.func @full(%x: tensor<2xf64>) -> tensor<5xf64> {
  %last = stablehlo.slice %x [1:2] : (tensor<2xf64>) -> tensor<1xf64>
  %w = "enzymexla.wrap"(%x) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 0 : i64}> : (tensor<2xf64>) -> tensor<4xf64>
  %c = stablehlo.concatenate %last, %w, dim = 0 : (tensor<1xf64>, tensor<4xf64>) -> tensor<5xf64>
  return %c : tensor<5xf64>
}

// CHECK:  func.func @full(%arg0: tensor<2xf64>) -> tensor<5xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 0 : i64}> : (tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<4xf64>) -> tensor<5xf64>
// CHECK-NEXT:   return %2 : tensor<5xf64>
// CHECK-NEXT: }

// -----

// With room left, the slice folds in.
func.func @room(%x: tensor<2xf64>) -> tensor<4xf64> {
  %first = stablehlo.slice %x [0:1] : (tensor<2xf64>) -> tensor<1xf64>
  %w = "enzymexla.wrap"(%x) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<2xf64>) -> tensor<3xf64>
  %c = stablehlo.concatenate %first, %w, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
  return %c : tensor<4xf64>
}

// CHECK:  func.func @room(%arg0: tensor<2xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %0 = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 0 : i64}> : (tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %0, dim = 0 : (tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %1 : tensor<4xf64>
// CHECK-NEXT: }
