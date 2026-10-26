// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=widen_extend;lower_extend" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// An extend repeats the first `lhs` elements before its operand: a boundary
// element next to an extend that already repeats one may not fold into it,
// since [x0, x0, x0, x1] is not extend(x, 2) = [x0, x1, x0, x1].
func.func @already_repeats(%x: tensor<2xf64>) -> tensor<4xf64> {
  %first = stablehlo.slice %x [0:1] : (tensor<2xf64>) -> tensor<1xf64>
  %e = "enzymexla.extend"(%x) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<2xf64>) -> tensor<3xf64>
  %c = stablehlo.concatenate %first, %e, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
  return %c : tensor<4xf64>
}

// CHECK:  func.func @already_repeats(%arg0: tensor<2xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %1, %arg0, dim = 0 : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
// CHECK-NEXT:   %3 = stablehlo.concatenate %0, %2, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %3 : tensor<4xf64>
// CHECK-NEXT: }

// -----

// With nothing repeated yet, the boundary element is the first one to be.
func.func @first_repeat(%x: tensor<2xf64>) -> tensor<3xf64> {
  %first = stablehlo.slice %x [0:1] : (tensor<2xf64>) -> tensor<1xf64>
  %e = "enzymexla.extend"(%x) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 0 : i64}> : (tensor<2xf64>) -> tensor<2xf64>
  %c = stablehlo.concatenate %first, %e, dim = 0 : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
  return %c : tensor<3xf64>
}

// CHECK:  func.func @first_repeat(%arg0: tensor<2xf64>) -> tensor<3xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %0, %arg0, dim = 0 : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %1, dim = 0 : (tensor<3xf64>) -> tensor<3xf64>
// CHECK-NEXT:   return %2 : tensor<3xf64>
// CHECK-NEXT: }
