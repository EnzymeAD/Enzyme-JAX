// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_operand_barrier" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The computation behind each operand of a concatenate of three or more
// operands is materialized: a sum read through a slice and a reshape, a
// product read through a broadcast; an argument, a constant and a value
// already behind a barrier are left alone, and a barrier serves every
// operand that reads the same value.
func.func @wide(%a: tensor<64x8xf64>, %b: tensor<64x8xf64>, %c: tensor<64x4xf64>) -> tensor<64x24xf64> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x4xf64>
  %0 = stablehlo.add %a, %b : tensor<64x8xf64>
  %1 = stablehlo.slice %0 [0:64, 0:4] : (tensor<64x8xf64>) -> tensor<64x4xf64>
  %2 = stablehlo.slice %0 [0:64, 4:8] : (tensor<64x8xf64>) -> tensor<64x4xf64>
  %3 = stablehlo.multiply %c, %c : tensor<64x4xf64>
  %4 = stablehlo.reshape %3 : (tensor<64x4xf64>) -> tensor<64x4x1xf64>
  %5 = stablehlo.reshape %4 : (tensor<64x4x1xf64>) -> tensor<64x4xf64>
  %6 = stablehlo.negate %c : tensor<64x4xf64>
  %7 = stablehlo.optimization_barrier %6 : tensor<64x4xf64>
  %8 = stablehlo.concatenate %1, %2, %5, %c, %cst, %7, dim = 1 : (tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>) -> tensor<64x24xf64>
  return %8 : tensor<64x24xf64>
}

// CHECK: func.func @wide(%arg0: tensor<64x8xf64>, %arg1: tensor<64x8xf64>, %arg2: tensor<64x4xf64>) -> tensor<64x24xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x4xf64>
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg1 : tensor<64x8xf64>
// CHECK-NEXT:   %1 = stablehlo.optimization_barrier %0 : tensor<64x8xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %1 [0:64, 0:4] : (tensor<64x8xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %3 = stablehlo.slice %1 [0:64, 4:8] : (tensor<64x8xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %4 = stablehlo.multiply %arg2, %arg2 : tensor<64x4xf64>
// CHECK-NEXT:   %5 = stablehlo.optimization_barrier %4 : tensor<64x4xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<64x4xf64>) -> tensor<64x4x1xf64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<64x4x1xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %8 = stablehlo.negate %arg2 : tensor<64x4xf64>
// CHECK-NEXT:   %9 = stablehlo.optimization_barrier %8 : tensor<64x4xf64>
// CHECK-NEXT:   %10 = stablehlo.concatenate %2, %3, %7, %arg2, %cst, %9, dim = 1 : (tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>) -> tensor<64x24xf64>
// CHECK-NEXT:   return %10 : tensor<64x24xf64>
// CHECK-NEXT: }

// A concatenate of two operands is not wide.
func.func @narrow(%a: tensor<64x8xf64>, %b: tensor<64x8xf64>) -> tensor<64x16xf64> {
  %0 = stablehlo.add %a, %b : tensor<64x8xf64>
  %1 = stablehlo.multiply %a, %b : tensor<64x8xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<64x8xf64>, tensor<64x8xf64>) -> tensor<64x16xf64>
  return %2 : tensor<64x16xf64>
}

// CHECK: func.func @narrow(%arg0: tensor<64x8xf64>, %arg1: tensor<64x8xf64>) -> tensor<64x16xf64> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg1 : tensor<64x8xf64>
// CHECK-NEXT:   %1 = stablehlo.multiply %arg0, %arg1 : tensor<64x8xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<64x8xf64>, tensor<64x8xf64>) -> tensor<64x16xf64>
// CHECK-NEXT:   return %2 : tensor<64x16xf64>
// CHECK-NEXT: }
