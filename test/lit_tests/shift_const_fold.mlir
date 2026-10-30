// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

// Shifts of constants fold.
func.func @shift_left_fold() -> tensor<4xi32> {
  %c = stablehlo.constant dense<[1, 2, -4, 8]> : tensor<4xi32>
  %s = stablehlo.constant dense<1> : tensor<4xi32>
  %0 = stablehlo.shift_left %c, %s : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK:      func.func @shift_left_fold() -> tensor<4xi32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[2, 4, -8, 16]> : tensor<4xi32>
// CHECK-NEXT:   return %c : tensor<4xi32>
// CHECK-NEXT: }

// -----


func.func @shift_right_arithmetic_fold() -> tensor<4xi32> {
  %c = stablehlo.constant dense<[1, 2, -4, 8]> : tensor<4xi32>
  %s = stablehlo.constant dense<1> : tensor<4xi32>
  %0 = stablehlo.shift_right_arithmetic %c, %s : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK:      func.func @shift_right_arithmetic_fold() -> tensor<4xi32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[0, 1, -2, 4]> : tensor<4xi32>
// CHECK-NEXT:   return %c : tensor<4xi32>
// CHECK-NEXT: }

// -----


func.func @shift_right_logical_fold() -> tensor<4xi32> {
  %c = stablehlo.constant dense<[1, 2, -4, 8]> : tensor<4xi32>
  %s = stablehlo.constant dense<1> : tensor<4xi32>
  %0 = stablehlo.shift_right_logical %c, %s : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK:      func.func @shift_right_logical_fold() -> tensor<4xi32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[0, 1, 2147483646, 4]> : tensor<4xi32>
// CHECK-NEXT:   return %c : tensor<4xi32>
// CHECK-NEXT: }
