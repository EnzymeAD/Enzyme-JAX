// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcast_in_dim_op_canon<16>" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A unit operand dimension the broadcast expands may map to any expanded
// output dimension; which one is a choice of spelling that tells
// broadcasts of the same scalars apart. The expanding unit dimensions
// take, in operand order, the positions they occupy in ascending order
// (cse then merges the two).
func.func @scalar(%s: tensor<1x1xf64>) -> (tensor<8x4x4xf64>, tensor<8x4x4xf64>) {
  %0 = stablehlo.broadcast_in_dim %s, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %s, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  return %0, %1 : tensor<8x4x4xf64>, tensor<8x4x4xf64>
}

// CHECK-LABEL: func.func @scalar(
// CHECK-SAME: %[[S:[^ ,:)]+]]: tensor<1x1xf64>) -> (tensor<8x4x4xf64>, tensor<8x4x4xf64>) {
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S]], dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[B1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S]], dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: return %[[B0]], %[[B1]] : tensor<8x4x4xf64>, tensor<8x4x4xf64>
// CHECK-NEXT: }

// A unit operand dimension landing on a unit output dimension is not
// expanded: its position is meaningful and stays.
func.func @kept_unit(%s: tensor<1x5xf64>) -> tensor<8x5x1xf64> {
  %0 = stablehlo.broadcast_in_dim %s, dims = [2, 1] : (tensor<1x5xf64>) -> tensor<8x5x1xf64>
  return %0 : tensor<8x5x1xf64>
}

// CHECK-LABEL: func.func @kept_unit(
// CHECK-SAME: %[[S:[^ ,:)]+]]: tensor<1x5xf64>) -> tensor<8x5x1xf64> {
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S]], dims = [2, 1] : (tensor<1x5xf64>) -> tensor<8x5x1xf64>
// CHECK-NEXT: return %[[B]] : tensor<8x5x1xf64>
// CHECK-NEXT: }
