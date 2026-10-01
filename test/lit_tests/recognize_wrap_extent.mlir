// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=recognize_wrap" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The pieces around a one-element window of the same tensor are longer than
// the window: a wrap takes its amounts from inside its operand, so this is
// not one.
func.func @too_long(%x: tensor<8xf64>) -> tensor<8xf64> {
  %a = stablehlo.slice %x [0:2] : (tensor<8xf64>) -> tensor<2xf64>
  %m = stablehlo.slice %x [2:3] : (tensor<8xf64>) -> tensor<1xf64>
  %b = stablehlo.slice %x [3:8] : (tensor<8xf64>) -> tensor<5xf64>
  %c = stablehlo.concatenate %a, %m, %b, dim = 0 : (tensor<2xf64>, tensor<1xf64>, tensor<5xf64>) -> tensor<8xf64>
  return %c : tensor<8xf64>
}

// CHECK:  func.func @too_long(%arg0: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NOT: enzymexla.wrap

// -----

// Pieces that fit are a wrap.
func.func @fits(%x: tensor<8xf64>) -> tensor<7xf64> {
  %a = stablehlo.slice %x [4:6] : (tensor<8xf64>) -> tensor<2xf64>
  %m = stablehlo.slice %x [2:6] : (tensor<8xf64>) -> tensor<4xf64>
  %b = stablehlo.slice %x [2:3] : (tensor<8xf64>) -> tensor<1xf64>
  %c = stablehlo.concatenate %a, %m, %b, dim = 0 : (tensor<2xf64>, tensor<4xf64>, tensor<1xf64>) -> tensor<7xf64>
  return %c : tensor<7xf64>
}

// CHECK:  func.func @fits(%arg0: tensor<8xf64>) -> tensor<7xf64> {
// CHECK:    "enzymexla.wrap"(%{{.*}}) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 1 : i64}>
