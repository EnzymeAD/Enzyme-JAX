// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=rem_simplify;div_simplify" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A lane index below the divisor: its remainder is itself, its quotient zero,
// through the transpose of the iota.
func.func @lane(%x: tensor<6x4xi32>) -> (tensor<4x6xi32>, tensor<4x6xi32>) {
  %c6 = stablehlo.constant dense<6> : tensor<4x6xi32>
  %i = stablehlo.iota dim = 0 : tensor<6x4xi32>
  %t = stablehlo.transpose %i, dims = [1, 0] : (tensor<6x4xi32>) -> tensor<4x6xi32>
  %r = stablehlo.remainder %t, %c6 : tensor<4x6xi32>
  %q = stablehlo.divide %t, %c6 : tensor<4x6xi32>
  return %r, %q : tensor<4x6xi32>, tensor<4x6xi32>
}

// CHECK:      func.func @lane(%arg0: tensor<6x4xi32>) -> (tensor<4x6xi32>, tensor<4x6xi32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<4x6xi32>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<6x4xi32>
// CHECK-NEXT:   %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<6x4xi32>) -> tensor<4x6xi32>
// CHECK-NEXT:   return %1, %c : tensor<4x6xi32>, tensor<4x6xi32>
// CHECK-NEXT: }

// -----


// An iota as long as the divisor, and a value that may be negative: kept.
func.func @kept(%x: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %c6 = stablehlo.constant dense<6> : tensor<8xi32>
  %c7 = stablehlo.constant dense<7> : tensor<8xi32>
  %i = stablehlo.iota dim = 0 : tensor<8xi32>
  %r = stablehlo.remainder %i, %c7 : tensor<8xi32>
  %q = stablehlo.divide %x, %c6 : tensor<8xi32>
  return %r, %q : tensor<8xi32>, tensor<8xi32>
}

// CHECK:      func.func @kept(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<6> : tensor<8xi32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<7> : tensor<8xi32>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi32>
// CHECK-NEXT:   %1 = stablehlo.remainder %0, %c_0 : tensor<8xi32>
// CHECK-NEXT:   %2 = stablehlo.divide %arg0, %c : tensor<8xi32>
// CHECK-NEXT:   return %1, %2 : tensor<8xi32>, tensor<8xi32>
// CHECK-NEXT: }
