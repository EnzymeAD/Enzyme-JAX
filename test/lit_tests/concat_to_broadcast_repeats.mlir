// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_to_broadcast" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The nested concat(a, a, a) made a broadcast first, then the outer concat.
func.func @nested_first(%b: tensor<5x5xf64>) -> tensor<5x5x5xf64> {
  %a = stablehlo.reshape %b : (tensor<5x5xf64>) -> tensor<1x5x5xf64>
  %c = stablehlo.broadcast_in_dim %b, dims = [1, 2] : (tensor<5x5xf64>) -> tensor<3x5x5xf64>
  %0 = stablehlo.concatenate %a, %a, %c, dim = 0 : (tensor<1x5x5xf64>, tensor<1x5x5xf64>, tensor<3x5x5xf64>) -> tensor<5x5x5xf64>
  return %0 : tensor<5x5x5xf64>
}

// CHECK-LABEL: func.func @nested_first(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x5xf64>) -> tensor<5x5x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [1, 2] : (tensor<5x5xf64>) -> tensor<5x5x5xf64>
// CHECK-NEXT: return %[[V1]] : tensor<5x5x5xf64>
// CHECK-NEXT: }

// a one wide along the concatenated dimension, broadcast along it in place.
func.func @same_rank(%a: tensor<4x1x6xf64>) -> tensor<4x6x6xf64> {
  %c = stablehlo.broadcast_in_dim %a, dims = [0, 1, 2] : (tensor<4x1x6xf64>) -> tensor<4x3x6xf64>
  %0 = stablehlo.concatenate %a, %c, %a, %a, dim = 1 : (tensor<4x1x6xf64>, tensor<4x3x6xf64>, tensor<4x1x6xf64>, tensor<4x1x6xf64>) -> tensor<4x6x6xf64>
  return %0 : tensor<4x6x6xf64>
}

// CHECK-LABEL: func.func @same_rank(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x1x6xf64>) -> tensor<4x6x6xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 1, 2] : (tensor<4x1x6xf64>) -> tensor<4x6x6xf64>
// CHECK-NEXT: return %[[V1]] : tensor<4x6x6xf64>
// CHECK-NEXT: }

// Stays: two different values.
func.func @different_values(%a: tensor<1x5xf64>, %b: tensor<1x5xf64>) -> tensor<3x5xf64> {
  %c = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x5xf64>) -> tensor<2x5xf64>
  %0 = stablehlo.concatenate %c, %b, dim = 0 : (tensor<2x5xf64>, tensor<1x5xf64>) -> tensor<3x5xf64>
  return %0 : tensor<3x5xf64>
}

// CHECK-LABEL: func.func @different_values(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1x5xf64>, %[[V1:[^ ,:)]+]]: tensor<1x5xf64>) -> tensor<3x5xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 1] : (tensor<1x5xf64>) -> tensor<2x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V1]], dim = 0 : (tensor<2x5xf64>, tensor<1x5xf64>) -> tensor<3x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3x5xf64>
// CHECK-NEXT: }

// Broadcasts that also widen another dimension, the same way.
func.func @widens_other_dim(%a: tensor<1x1xf64>) -> tensor<3x5xf64> {
  %c = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<2x5xf64>
  %d = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x5xf64>
  %0 = stablehlo.concatenate %c, %d, dim = 0 : (tensor<2x5xf64>, tensor<1x5xf64>) -> tensor<3x5xf64>
  return %0 : tensor<3x5xf64>
}

// CHECK-LABEL: func.func @widens_other_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1x1xf64>) -> tensor<3x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 1] : (tensor<1x1xf64>) -> tensor<3x5xf64>
// CHECK-NEXT: return %[[V1]] : tensor<3x5xf64>
// CHECK-NEXT: }

// Only the adjacent run of one value merges.
func.func @run(%x: tensor<2x5xf64>, %a: tensor<1x5xf64>, %y: tensor<1x5xf64>) -> tensor<7x5xf64> {
  %c = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x5xf64>) -> tensor<2x5xf64>
  %0 = stablehlo.concatenate %x, %a, %c, %a, %y, dim = 0 : (tensor<2x5xf64>, tensor<1x5xf64>, tensor<2x5xf64>, tensor<1x5xf64>, tensor<1x5xf64>) -> tensor<7x5xf64>
  return %0 : tensor<7x5xf64>
}

// CHECK-LABEL: func.func @run(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x5xf64>, %[[V1:[^ ,:)]+]]: tensor<1x5xf64>, %[[V2:[^ ,:)]+]]: tensor<1x5xf64>) -> tensor<7x5xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1] : (tensor<1x5xf64>) -> tensor<4x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V3]], %[[V2]], dim = 0 : (tensor<2x5xf64>, tensor<4x5xf64>, tensor<1x5xf64>) -> tensor<7x5xf64>
// CHECK-NEXT: return %[[V4]] : tensor<7x5xf64>
// CHECK-NEXT: }

// A value two wide along the concatenated dimension repeats as a block.
func.func @wide_value(%a: tensor<2x1xf64>) -> tensor<4x5xf64> {
  %c = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<2x1xf64>) -> tensor<2x5xf64>
  %0 = stablehlo.concatenate %c, %c, dim = 0 : (tensor<2x5xf64>, tensor<2x5xf64>) -> tensor<4x5xf64>
  return %0 : tensor<4x5xf64>
}

// CHECK-LABEL: func.func @wide_value(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x1xf64>) -> tensor<4x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 1] : (tensor<2x1xf64>) -> tensor<2x5xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<2x5xf64>) -> tensor<1x2x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [0, 1, 2] : (tensor<1x2x5xf64>) -> tensor<2x2x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<2x2x5xf64>) -> tensor<4x5xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4x5xf64>
// CHECK-NEXT: }

// Stays: the same value broadcast to different dimensions.
func.func @different_dims(%a: tensor<1x5xf64>) -> tensor<2x5x5xf64> {
  %c = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x5xf64>) -> tensor<1x5x5xf64>
  %d = stablehlo.broadcast_in_dim %a, dims = [0, 2] : (tensor<1x5xf64>) -> tensor<1x5x5xf64>
  %0 = stablehlo.concatenate %c, %d, dim = 0 : (tensor<1x5x5xf64>, tensor<1x5x5xf64>) -> tensor<2x5x5xf64>
  return %0 : tensor<2x5x5xf64>
}

// CHECK-LABEL: func.func @different_dims(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1x5xf64>) -> tensor<2x5x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 1] : (tensor<1x5xf64>) -> tensor<1x5x5xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [0, 2] : (tensor<1x5xf64>) -> tensor<1x5x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V1]], %[[V2]], dim = 0 : (tensor<1x5x5xf64>, tensor<1x5x5xf64>) -> tensor<2x5x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<2x5x5xf64>
// CHECK-NEXT: }
