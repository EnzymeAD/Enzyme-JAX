// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape_block_align;concat_blocked_windows" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Windows of one buffer stacked at different times: a single window read
// through the merged reshape ([N, 75, 5], the first 25 rows, then split into
// the block's shape and given a unit dimension) and a run of two windows
// read through the blocked reshape ([N, 3, 25, 5], blocks 1 and 2, through a
// transpose). slice_reshape_block_align reads the single window through the
// blocked reshape too, and concat_blocked_windows then takes the run of
// three blocks as one slice (here the whole buffer) with the run's views.
func.func @straggler(%a: tensor<3472875xf64>) -> tensor<3x5x9261x5x5xf64> {
  %3 = stablehlo.reshape %a : (tensor<3472875xf64>) -> tensor<9261x75x5xf64>
  %7 = stablehlo.slice %3 [0:9261, 0:25, 0:5] : (tensor<9261x75x5xf64>) -> tensor<9261x25x5xf64>
  %8 = stablehlo.reshape %7 : (tensor<9261x25x5xf64>) -> tensor<9261x5x5x5xf64>
  %16 = stablehlo.broadcast_in_dim %8, dims = [2, 1, 3, 4] : (tensor<9261x5x5x5xf64>) -> tensor<1x5x9261x5x5xf64>
  %11 = stablehlo.reshape %a : (tensor<3472875xf64>) -> tensor<9261x3x25x5xf64>
  %12 = stablehlo.transpose %11, dims = [1, 0, 2, 3] : (tensor<9261x3x25x5xf64>) -> tensor<3x9261x25x5xf64>
  %13 = stablehlo.slice %12 [1:3, 0:9261, 0:25, 0:5] : (tensor<3x9261x25x5xf64>) -> tensor<2x9261x25x5xf64>
  %14 = stablehlo.reshape %13 : (tensor<2x9261x25x5xf64>) -> tensor<2x9261x5x5x5xf64>
  %15 = stablehlo.transpose %14, dims = [0, 2, 1, 3, 4] : (tensor<2x9261x5x5x5xf64>) -> tensor<2x5x9261x5x5xf64>
  %17 = stablehlo.concatenate %16, %15, dim = 0 : (tensor<1x5x9261x5x5xf64>, tensor<2x5x9261x5x5xf64>) -> tensor<3x5x9261x5x5xf64>
  return %17 : tensor<3x5x9261x5x5xf64>
}

// CHECK-LABEL: func.func @straggler(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<3472875xf64>) -> tensor<3x5x9261x5x5xf64> {
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<3472875xf64>) -> tensor<9261x3x25x5xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[B]], dims = [1, 0, 2, 3] : (tensor<9261x3x25x5xf64>) -> tensor<3x9261x25x5xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[T]] : (tensor<3x9261x25x5xf64>) -> tensor<3x9261x5x5x5xf64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [0, 2, 1, 3, 4] : (tensor<3x9261x5x5x5xf64>) -> tensor<3x5x9261x5x5xf64>
// CHECK-NEXT: return %[[U]] : tensor<3x5x9261x5x5xf64>
// CHECK-NEXT: }

// A run of two consecutive slices of one value, the single one behind a
// unit reshape: the slices read the same value, so this is concat_slice's
// and left alone here.
func.func @same_operand(%x: tensor<3x8xf64>) -> tensor<3x8xf64> {
  %0 = stablehlo.slice %x [0:1, 0:8] : (tensor<3x8xf64>) -> tensor<1x8xf64>
  %1 = stablehlo.reshape %0 : (tensor<1x8xf64>) -> tensor<1x8xf64>
  %2 = stablehlo.slice %x [1:3, 0:8] : (tensor<3x8xf64>) -> tensor<2x8xf64>
  %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<1x8xf64>, tensor<2x8xf64>) -> tensor<3x8xf64>
  return %3 : tensor<3x8xf64>
}

// CHECK-LABEL: func.func @same_operand(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<3x8xf64>) -> tensor<3x8xf64> {
// CHECK-NEXT: %[[S0:[^ ,:)]+]] = stablehlo.slice %[[X]] [0:1, 0:8] : (tensor<3x8xf64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reshape %[[S0]] : (tensor<1x8xf64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.slice %[[X]] [1:3, 0:8] : (tensor<3x8xf64>) -> tensor<2x8xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[R0]], %[[S2]], dim = 0 : (tensor<1x8xf64>, tensor<2x8xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: return %[[C]] : tensor<3x8xf64>
// CHECK-NEXT: }
