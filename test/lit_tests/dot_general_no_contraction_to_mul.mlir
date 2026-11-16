// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_no_contraction_to_mul" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Result dims: batch (x dim 0 / y dim 1), then x free dim 1, then y free dim 0.
func.func @batched_outer(%x: tensor<5x4xf64>, %y: tensor<3x5xf64>) -> tensor<5x4x3xf64> {
  %0 = stablehlo.dot_general %x, %y, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<5x4xf64>, tensor<3x5xf64>) -> tensor<5x4x3xf64>
  return %0 : tensor<5x4x3xf64>
}
// CHECK-LABEL: func.func @batched_outer
// CHECK-DAG:  %[[X:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<5x4xf64>) -> tensor<5x4x3xf64>
// CHECK-DAG:  %[[Y:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [2, 0] : (tensor<3x5xf64>) -> tensor<5x4x3xf64>
// CHECK:      %[[R:.+]] = stablehlo.multiply %[[X]], %[[Y]] : tensor<5x4x3xf64>
// CHECK-NEXT: return %[[R]]

// A dot that contracts a dimension is left alone.
func.func @contracting(%x: tensor<5x4xf64>, %y: tensor<4x3xf64>) -> tensor<5x3xf64> {
  %0 = stablehlo.dot_general %x, %y, contracting_dims = [1] x [0] : (tensor<5x4xf64>, tensor<4x3xf64>) -> tensor<5x3xf64>
  return %0 : tensor<5x3xf64>
}
// CHECK-LABEL: func.func @contracting
// CHECK:      stablehlo.dot_general

// A plain outer product (no batching dimension) is left alone.
func.func @outer(%x: tensor<5xf64>, %y: tensor<3xf64>) -> tensor<5x3xf64> {
  %0 = stablehlo.dot_general %x, %y, contracting_dims = [] x [] : (tensor<5xf64>, tensor<3xf64>) -> tensor<5x3xf64>
  return %0 : tensor<5x3xf64>
}
// CHECK-LABEL: func.func @outer
// CHECK:      stablehlo.dot_general

// A non-default precision request is kept on the dot.
func.func @precision(%x: tensor<5x4xf32>, %y: tensor<3x5xf32>) -> tensor<5x4x3xf32> {
  %0 = stablehlo.dot_general %x, %y, batching_dims = [0] x [1], contracting_dims = [] x [], precision = [HIGHEST, HIGHEST] : (tensor<5x4xf32>, tensor<3x5xf32>) -> tensor<5x4x3xf32>
  return %0 : tensor<5x4x3xf32>
}
// CHECK-LABEL: func.func @precision
// CHECK:      stablehlo.dot_general
