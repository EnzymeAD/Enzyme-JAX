// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=select_comp_iota_const_simplify" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// `iota == N` on a dimension of size N is never true: the select is its
// false operand, not a slice ending past the dimension.
func.func @eq_end(%a: tensor<158x80xf64>, %b: tensor<158x80xf64>) -> tensor<158x80xf64> {
  %iota = stablehlo.iota dim = 0 : tensor<158x80xi32>
  %k = stablehlo.constant dense<158> : tensor<158x80xi32>
  %p = stablehlo.compare EQ, %iota, %k : (tensor<158x80xi32>, tensor<158x80xi32>) -> tensor<158x80xi1>
  %r = stablehlo.select %p, %a, %b : tensor<158x80xi1>, tensor<158x80xf64>
  return %r : tensor<158x80xf64>
}

// CHECK:  func.func @eq_end(%arg0: tensor<158x80xf64>, %arg1: tensor<158x80xf64>) -> tensor<158x80xf64> {
// CHECK-NEXT:   return %arg1 : tensor<158x80xf64>
// CHECK-NEXT: }

// -----

// `iota != N` is always true.
func.func @ne_end(%a: tensor<20xi64>, %b: tensor<20xi64>) -> tensor<20xi64> {
  %iota = stablehlo.iota dim = 0 : tensor<20xi64>
  %k = stablehlo.constant dense<20> : tensor<20xi64>
  %p = stablehlo.compare NE, %iota, %k : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
  %r = stablehlo.select %p, %a, %b : tensor<20xi1>, tensor<20xi64>
  return %r : tensor<20xi64>
}

// CHECK:  func.func @ne_end(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
// CHECK-NEXT:   return %arg0 : tensor<20xi64>
// CHECK-NEXT: }

// -----

// A bound beyond the dimension on either side leaves the predicate uniform.
func.func @lt_past_end(%a: tensor<20xi64>, %b: tensor<20xi64>) -> tensor<20xi64> {
  %iota = stablehlo.iota dim = 0 : tensor<20xi64>
  %k = stablehlo.constant dense<25> : tensor<20xi64>
  %p = stablehlo.compare LT, %iota, %k : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
  %r = stablehlo.select %p, %a, %b : tensor<20xi1>, tensor<20xi64>
  return %r : tensor<20xi64>
}

// CHECK:  func.func @lt_past_end(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
// CHECK-NEXT:   return %arg0 : tensor<20xi64>
// CHECK-NEXT: }

// -----

func.func @gt_negative(%a: tensor<20xi64>, %b: tensor<20xi64>) -> tensor<20xi64> {
  %iota = stablehlo.iota dim = 0 : tensor<20xi64>
  %k = stablehlo.constant dense<-3> : tensor<20xi64>
  %p = stablehlo.compare GT, %iota, %k : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
  %r = stablehlo.select %p, %a, %b : tensor<20xi1>, tensor<20xi64>
  return %r : tensor<20xi64>
}

// CHECK:  func.func @gt_negative(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
// CHECK-NEXT:   return %arg0 : tensor<20xi64>
// CHECK-NEXT: }

// -----

// `iota == N - 1` keeps its last-row slice.
func.func @eq_last(%a: tensor<20xi64>, %b: tensor<20xi64>) -> tensor<20xi64> {
  %iota = stablehlo.iota dim = 0 : tensor<20xi64>
  %k = stablehlo.constant dense<19> : tensor<20xi64>
  %p = stablehlo.compare EQ, %iota, %k : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
  %r = stablehlo.select %p, %a, %b : tensor<20xi1>, tensor<20xi64>
  return %r : tensor<20xi64>
}

// CHECK:  func.func @eq_last(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:19] : (tensor<20xi64>) -> tensor<19xi64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [19:20] : (tensor<20xi64>) -> tensor<1xi64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<19xi64>, tensor<1xi64>) -> tensor<20xi64>
// CHECK-NEXT:   return %2 : tensor<20xi64>
// CHECK-NEXT: }

// -----

// A unit iota dimension broadcast along the output dimension is constant
// there, so the select is not a split of that dimension.
func.func @unit_iota(%a: tensor<20xi64>, %b: tensor<20xi64>) -> tensor<20xi64> {
  %iota = stablehlo.iota dim = 0 : tensor<1xi64>
  %k = stablehlo.constant dense<5> : tensor<1xi64>
  %p = stablehlo.compare LT, %iota, %k : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
  %pb = stablehlo.broadcast_in_dim %p, dims = [0] : (tensor<1xi1>) -> tensor<20xi1>
  %r = stablehlo.select %pb, %a, %b : tensor<20xi1>, tensor<20xi64>
  return %r : tensor<20xi64>
}

// CHECK:  func.func @unit_iota(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
// CHECK-NOT: stablehlo.slice
// CHECK:   stablehlo.select
