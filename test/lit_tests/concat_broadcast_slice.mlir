// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_broadcast_slice" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s

// The sliced dimension maps onto the concat dimension: merging is legal.
func.func @merge(%arg0: tensor<65xf64>) -> tensor<10x1xf64> {
  %0 = stablehlo.slice %arg0 [0:5] : (tensor<65xf64>) -> tensor<5xf64>
  %1 = stablehlo.reshape %0 : (tensor<5xf64>) -> tensor<5x1xf64>
  %2 = stablehlo.slice %arg0 [5:10] : (tensor<65xf64>) -> tensor<5xf64>
  %3 = stablehlo.reshape %2 : (tensor<5xf64>) -> tensor<5x1xf64>
  %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<5x1xf64>, tensor<5x1xf64>) -> tensor<10x1xf64>
  return %4 : tensor<10x1xf64>
}

// CHECK-LABEL: func.func @merge
// CHECK-NEXT:    %[[SLICE:.+]] = stablehlo.slice %arg0 [0:10] : (tensor<65xf64>) -> tensor<10xf64>
// CHECK-NEXT:    %[[BCAST:.+]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [0] : (tensor<10xf64>) -> tensor<10x1xf64>
// CHECK-NEXT:    %[[CONCAT:.+]] = stablehlo.concatenate %[[BCAST]], dim = 0
// CHECK-NEXT:    return %[[CONCAT]]

// -----

// The slice happens on dim 0 while the concat is on dim 1: merging the slices
// would grow the wrong dimension, so this must be left alone.
func.func @wrong_dim(%arg0: tensor<65xf64>) -> tensor<5x3xf64> {
  %0 = stablehlo.reshape %arg0 : (tensor<65xf64>) -> tensor<65x1xf64>
  %1 = stablehlo.slice %0 [0:5, 0:1] : (tensor<65x1xf64>) -> tensor<5x1xf64>
  %2 = stablehlo.slice %0 [5:10, 0:1] : (tensor<65x1xf64>) -> tensor<5x1xf64>
  %3 = stablehlo.slice %0 [10:15, 0:1] : (tensor<65x1xf64>) -> tensor<5x1xf64>
  %4 = stablehlo.concatenate %1, %2, %3, dim = 1 : (tensor<5x1xf64>, tensor<5x1xf64>, tensor<5x1xf64>) -> tensor<5x3xf64>
  return %4 : tensor<5x3xf64>
}

// CHECK-LABEL: func.func @wrong_dim
// CHECK-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %arg0
// CHECK-NEXT:    %[[S0:.+]] = stablehlo.slice %[[RESHAPE]] [0:5, 0:1]
// CHECK-NEXT:    %[[S1:.+]] = stablehlo.slice %[[RESHAPE]] [5:10, 0:1]
// CHECK-NEXT:    %[[S2:.+]] = stablehlo.slice %[[RESHAPE]] [10:15, 0:1]
// CHECK-NEXT:    %[[CONCAT:.+]] = stablehlo.concatenate %[[S0]], %[[S1]], %[[S2]], dim = 1
// CHECK-NEXT:    return %[[CONCAT]]

// -----

// Same, but the sliced dim maps onto the concat dim through a reshape.
func.func @merge_through_reshape(%arg0: tensor<65xf64>) -> tensor<1x10xf64> {
  %0 = stablehlo.slice %arg0 [0:5] : (tensor<65xf64>) -> tensor<5xf64>
  %1 = stablehlo.reshape %0 : (tensor<5xf64>) -> tensor<1x5xf64>
  %2 = stablehlo.slice %arg0 [5:10] : (tensor<65xf64>) -> tensor<5xf64>
  %3 = stablehlo.reshape %2 : (tensor<5xf64>) -> tensor<1x5xf64>
  %4 = stablehlo.concatenate %1, %3, dim = 1 : (tensor<1x5xf64>, tensor<1x5xf64>) -> tensor<1x10xf64>
  return %4 : tensor<1x10xf64>
}

// CHECK-LABEL: func.func @merge_through_reshape
// CHECK-NEXT:    %[[SLICE:.+]] = stablehlo.slice %arg0 [0:10] : (tensor<65xf64>) -> tensor<10xf64>
// CHECK-NEXT:    %[[BCAST:.+]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [1] : (tensor<10xf64>) -> tensor<1x10xf64>
// CHECK-NEXT:    %[[CONCAT:.+]] = stablehlo.concatenate %[[BCAST]], dim = 1
// CHECK-NEXT:    return %[[CONCAT]]

// -----

// Both operands slice dim 0 (the concat dim), but they disagree on where the
// remaining source dims land: %1 keeps them, %2 swaps them. The merged value
// can only carry one of the two layouts, so this must be left alone.
func.func @perm_mismatch(%arg0: tensor<4x5x5xf64>) -> tensor<4x5x5xf64> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<4x5x5xf64>) -> tensor<4x5x5xf64>
  %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1] : (tensor<4x5x5xf64>) -> tensor<4x5x5xf64>
  %2 = stablehlo.slice %0 [0:2, 0:5, 0:5] : (tensor<4x5x5xf64>) -> tensor<2x5x5xf64>
  %3 = stablehlo.slice %1 [2:4, 0:5, 0:5] : (tensor<4x5x5xf64>) -> tensor<2x5x5xf64>
  %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<2x5x5xf64>, tensor<2x5x5xf64>) -> tensor<4x5x5xf64>
  return %4 : tensor<4x5x5xf64>
}

// CHECK-LABEL: func.func @perm_mismatch
// CHECK-NEXT:    %[[B0:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2]
// CHECK-NEXT:    %[[B1:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1]
// CHECK-NEXT:    %[[S0:.+]] = stablehlo.slice %[[B0]] [0:2, 0:5, 0:5]
// CHECK-NEXT:    %[[S1:.+]] = stablehlo.slice %[[B1]] [2:4, 0:5, 0:5]
// CHECK-NEXT:    %[[CONCAT:.+]] = stablehlo.concatenate %[[S0]], %[[S1]], dim = 0
// CHECK-NEXT:    return %[[CONCAT]]

// -----

// Broadcast expands a size-1 slice along the concat dim: peeling through the
// expanding broadcast would miscompute both the slice range and output size.
func.func @no_peel_expanding_broadcast_on_concat_dim(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  %s0 = stablehlo.slice %arg0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [0] : (tensor<1xf64>) -> tensor<4xf64>
  %c0 = stablehlo.slice %b0 [0:2] : (tensor<4xf64>) -> tensor<2xf64>
  %s1 = stablehlo.slice %arg0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
  %c1 = stablehlo.broadcast_in_dim %s1, dims = [0] : (tensor<1xf64>) -> tensor<2xf64>
  %cat = stablehlo.concatenate %c0, %c1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  return %cat : tensor<4xf64>
}

// CHECK-LABEL: func.func @no_peel_expanding_broadcast_on_concat_dim
// CHECK:         %[[CONCAT:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %[[CONCAT]]

// -----

// Non-concat unit dimension of the source is broadcast-expanded to 5: merging
// along dim 0 must preserve the expanded size 5 on dim 1.
func.func @preserve_expanded_non_concat_dim(%arg0: tensor<6x1xf64>) -> tensor<4x5xf64> {
  %s0 = stablehlo.slice %arg0 [0:2, 0:1] : (tensor<6x1xf64>) -> tensor<2x1xf64>
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [0, 1] : (tensor<2x1xf64>) -> tensor<2x5xf64>
  %s1 = stablehlo.slice %arg0 [2:4, 0:1] : (tensor<6x1xf64>) -> tensor<2x1xf64>
  %b1 = stablehlo.broadcast_in_dim %s1, dims = [0, 1] : (tensor<2x1xf64>) -> tensor<2x5xf64>
  %cat = stablehlo.concatenate %b0, %b1, dim = 0 : (tensor<2x5xf64>, tensor<2x5xf64>) -> tensor<4x5xf64>
  return %cat : tensor<4x5xf64>
}

// CHECK-LABEL: func.func @preserve_expanded_non_concat_dim
// CHECK-NEXT:    %[[SLICE:.+]] = stablehlo.slice %arg0 [0:4, 0:1] : (tensor<6x1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %[[BCAST:.+]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:    %[[CONCAT:.+]] = stablehlo.concatenate %[[BCAST]], dim = 0 : (tensor<4x5xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:    return %[[CONCAT]]
