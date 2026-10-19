// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=select_comp_iota_const_simplify" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A select under a constant mask that is a run of true then false along one
// dimension (`lane < 3`, the step of an unrolled tree reduction) is the
// concatenation of the operands' slices on either side of the split.
func.func @prefix(%a: tensor<8xf64>, %b: tensor<8xf64>) -> tensor<8xf64> {
  %m = stablehlo.constant dense<[true, true, true, false, false, false, false, false]> : tensor<8xi1>
  %r = stablehlo.select %m, %a, %b : tensor<8xi1>, tensor<8xf64>
  return %r : tensor<8xf64>
}

// CHECK:  func.func @prefix(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:3] : (tensor<8xf64>) -> tensor<3xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [3:8] : (tensor<8xf64>) -> tensor<5xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<3xf64>, tensor<5xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %2 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// The false run first picks the false operand first.
func.func @suffix(%a: tensor<8xf64>, %b: tensor<8xf64>) -> tensor<8xf64> {
  %m = stablehlo.constant dense<[false, false, true, true, true, true, true, true]> : tensor<8xi1>
  %r = stablehlo.select %m, %a, %b : tensor<8xi1>, tensor<8xf64>
  return %r : tensor<8xf64>
}

// CHECK:  func.func @suffix(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:2] : (tensor<8xf64>) -> tensor<2xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [2:8] : (tensor<8xf64>) -> tensor<6xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<6xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %2 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// The run may be along any dimension as long as the mask is uniform along
// the others: here columns 0 and 1 are true, 2 and 3 false.
func.func @columns(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %m = stablehlo.constant dense<[[true, true, false, false], [true, true, false, false]]> : tensor<2x4xi1>
  %r = stablehlo.select %m, %a, %b : tensor<2x4xi1>, tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// CHECK:  func.func @columns(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:2, 0:2] : (tensor<2x4xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [0:2, 2:4] : (tensor<2x4xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   return %2 : tensor<2x4xf32>
// CHECK-NEXT: }

// -----

// More than one change of value along the dimension: not a prefix, stays.
func.func @alternating(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4xf64> {
  %m = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %r = stablehlo.select %m, %a, %b : tensor<4xi1>, tensor<4xf64>
  return %r : tensor<4xf64>
}

// CHECK:  func.func @alternating(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK-NEXT:   %0 = stablehlo.select %c, %arg0, %arg1 : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:   return %0 : tensor<4xf64>
// CHECK-NEXT: }
