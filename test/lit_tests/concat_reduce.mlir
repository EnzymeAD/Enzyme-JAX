// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Three sums over dimension 2, concatenated along dimension 1 (input
// dimension 1): one sum of the concatenated inputs.
func.func @kept_dim(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>, %c: tensor<8x2x4xf64>) -> tensor<8x8xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rc = stablehlo.reduce(%c init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x2x4xf64>, tensor<f64>) -> tensor<8x2xf64>
  %r = stablehlo.concatenate %ra, %rb, %rc, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>, tensor<8x2xf64>) -> tensor<8x8xf64>
  return %r : tensor<8x8xf64>
}

// CHECK-LABEL: func.func @kept_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x2x4xf64>) -> tensor<8x8xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], %[[V2]], dim = 1 : (tensor<8x3x4xf64>, tensor<8x3x4xf64>, tensor<8x2x4xf64>) -> tensor<8x8x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reduce(%[[V4]] init: %[[V3]]) applies stablehlo.add across dimensions = [2] : (tensor<8x8x4xf64>, tensor<f64>) -> tensor<8x8xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x8xf64>
// CHECK-NEXT: }

// The reduced dimension comes first: result dimension 1 is input dimension 2.
func.func @after_reduced(%a: tensor<4x8x3xf64>, %b: tensor<4x8x3xf64>) -> tensor<8x6xf64> {
  %z = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8x3xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8x3xf64>, tensor<f64>) -> tensor<8x3xf64>
  %r = stablehlo.concatenate %ra, %rb, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
  return %r : tensor<8x6xf64>
}

// CHECK-LABEL: func.func @after_reduced(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x3xf64>, %[[V1:[^ ,:)]+]]: tensor<4x8x3xf64>) -> tensor<8x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 2 : (tensor<4x8x3xf64>, tensor<4x8x3xf64>) -> tensor<4x8x6xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V3]] init: %[[V2]]) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8x6xf64>, tensor<f64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x6xf64>
// CHECK-NEXT: }

// Different reductions (add and max): left alone.
func.func @different_body(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> tensor<8x6xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.maximum across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %r = stablehlo.concatenate %ra, %rb, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
  return %r : tensor<8x6xf64>
}

// CHECK-LABEL: func.func @different_body(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>) -> tensor<8x6xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V1]] init: %[[V2]]) applies stablehlo.maximum across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x6xf64>
// CHECK-NEXT: }

// A reduce also used on its own: left alone.
func.func @other_user(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> (tensor<8x6xf64>, tensor<8x3xf64>) {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %r = stablehlo.concatenate %ra, %rb, dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
  return %r, %ra : tensor<8x6xf64>, tensor<8x3xf64>
}

// CHECK-LABEL: func.func @other_user(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>) -> (tensor<8x6xf64>, tensor<8x3xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V1]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 1 : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %[[V5]], %[[V3]] : tensor<8x6xf64>, tensor<8x3xf64>
// CHECK-NEXT: }

// Reshapes that merge the concatenated dimension with the one after it: one
// sum of the inputs concatenated along input dimension 1, reshaped.
func.func @reshaped(%a: tensor<8x3x3x4xf64>, %b: tensor<8x3x3x4xf64>) -> tensor<8x18xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [3] : (tensor<8x3x3x4xf64>, tensor<f64>) -> tensor<8x3x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [3] : (tensor<8x3x3x4xf64>, tensor<f64>) -> tensor<8x3x3xf64>
  %sa = stablehlo.reshape %ra : (tensor<8x3x3xf64>) -> tensor<8x9xf64>
  %sb = stablehlo.reshape %rb : (tensor<8x3x3xf64>) -> tensor<8x9xf64>
  %r = stablehlo.concatenate %sa, %sb, dim = 1 : (tensor<8x9xf64>, tensor<8x9xf64>) -> tensor<8x18xf64>
  return %r : tensor<8x18xf64>
}

// CHECK-LABEL: func.func @reshaped(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x3x4xf64>) -> tensor<8x18xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 1 : (tensor<8x3x3x4xf64>, tensor<8x3x3x4xf64>) -> tensor<8x6x3x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V3]] init: %[[V2]]) applies stablehlo.add across dimensions = [3] : (tensor<8x6x3x4xf64>, tensor<f64>) -> tensor<8x6x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<8x6x3xf64>) -> tensor<8x18xf64>
// CHECK-NEXT: return %[[V5]] : tensor<8x18xf64>
// CHECK-NEXT: }

// Reshapes that only insert the concatenated dimension: left to
// concat_insert_dim_reduce.
func.func @inserted(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> tensor<2x8x3xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
  %sa = stablehlo.reshape %ra : (tensor<8x3xf64>) -> tensor<1x8x3xf64>
  %sb = stablehlo.reshape %rb : (tensor<8x3xf64>) -> tensor<1x8x3xf64>
  %r = stablehlo.concatenate %sa, %sb, dim = 0 : (tensor<1x8x3xf64>, tensor<1x8x3xf64>) -> tensor<2x8x3xf64>
  return %r : tensor<2x8x3xf64>
}

// CHECK-LABEL: func.func @inserted(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3x4xf64>) -> tensor<2x8x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V0]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V1]] init: %[[V2]]) applies stablehlo.add across dimensions = [2] : (tensor<8x3x4xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<8x3xf64>) -> tensor<1x8x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<8x3xf64>) -> tensor<1x8x3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.concatenate %[[V5]], %[[V6]], dim = 0 : (tensor<1x8x3xf64>, tensor<1x8x3xf64>) -> tensor<2x8x3xf64>
// CHECK-NEXT: return %[[V7]] : tensor<2x8x3xf64>
// CHECK-NEXT: }

// Reductions that differ only in a unit dimension they reduce (a broadcast
// inserted one in the second; delete_dims_reduce folds a unit-deleting
// reshape into a reduction): the same reduction, concatenated through with
// the unit dimension dropped.
func.func @unit_reduced(%x: tensor<8x4x3x3xf64>, %y: tensor<8x4x3x3x1xf64>) -> tensor<8x6x3xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x4x3x3xf64>, tensor<f64>) -> tensor<8x3x3xf64>
  %1 = stablehlo.reduce(%y init: %z) applies stablehlo.add across dimensions = [1, 4] : (tensor<8x4x3x3x1xf64>, tensor<f64>) -> tensor<8x3x3xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<8x3x3xf64>, tensor<8x3x3xf64>) -> tensor<8x6x3xf64>
  return %2 : tensor<8x6x3xf64>
}

// CHECK-LABEL: func.func @unit_reduced(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<8x4x3x3xf64>, %[[Y:[^ ,:)]+]]: tensor<8x4x3x3x1xf64>) -> tensor<8x6x3xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[YR:[^ ,:)]+]] = stablehlo.reshape %[[Y]] : (tensor<8x4x3x3x1xf64>) -> tensor<8x4x3x3xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[X]], %[[YR]], dim = 2 : (tensor<8x4x3x3xf64>, tensor<8x4x3x3xf64>) -> tensor<8x4x6x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[C]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x4x6x3xf64>, tensor<f64>) -> tensor<8x6x3xf64>
// CHECK-NEXT: return %[[R]] : tensor<8x6x3xf64>
// CHECK-NEXT: }

// Reductions stacked along a dimension their reshapes insert while
// deleting a unit result dimension (not a pure insertion, so not
// concat_insert_dim_reduce's): one reduction of the stacked inputs.
func.func @stacked_reshape(%a: tensor<3x8x4x3x1xf64>, %b: tensor<3x8x4x3x1xf64>, %c: tensor<3x8x4x3x1xf64>) -> tensor<3x8x4x3xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x3x1xf64>, tensor<f64>) -> tensor<8x4x3x1xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x3x1xf64>, tensor<f64>) -> tensor<8x4x3x1xf64>
  %rc = stablehlo.reduce(%c init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x3x1xf64>, tensor<f64>) -> tensor<8x4x3x1xf64>
  %sa = stablehlo.reshape %ra : (tensor<8x4x3x1xf64>) -> tensor<1x8x4x3xf64>
  %sb = stablehlo.reshape %rb : (tensor<8x4x3x1xf64>) -> tensor<1x8x4x3xf64>
  %sc = stablehlo.reshape %rc : (tensor<8x4x3x1xf64>) -> tensor<1x8x4x3xf64>
  %r = stablehlo.concatenate %sa, %sb, %sc, dim = 0 : (tensor<1x8x4x3xf64>, tensor<1x8x4x3xf64>, tensor<1x8x4x3xf64>) -> tensor<3x8x4x3xf64>
  return %r : tensor<3x8x4x3xf64>
}

// CHECK-LABEL: func.func @stacked_reshape(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<3x8x4x3x1xf64>, %[[B:[^ ,:)]+]]: tensor<3x8x4x3x1xf64>, %[[C:[^ ,:)]+]]: tensor<3x8x4x3x1xf64>) -> tensor<3x8x4x3xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[UA:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<3x8x4x3x1xf64>) -> tensor<1x3x8x4x3x1xf64>
// CHECK-NEXT: %[[UB:[^ ,:)]+]] = stablehlo.reshape %[[B]] : (tensor<3x8x4x3x1xf64>) -> tensor<1x3x8x4x3x1xf64>
// CHECK-NEXT: %[[UC:[^ ,:)]+]] = stablehlo.reshape %[[C]] : (tensor<3x8x4x3x1xf64>) -> tensor<1x3x8x4x3x1xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[UA]], %[[UB]], %[[UC]], dim = 0 : (tensor<1x3x8x4x3x1xf64>, tensor<1x3x8x4x3x1xf64>, tensor<1x3x8x4x3x1xf64>) -> tensor<3x3x8x4x3x1xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[S]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x3x1xf64>, tensor<f64>) -> tensor<3x8x4x3x1xf64>
// CHECK-NEXT: %[[V:[^ ,:)]+]] = stablehlo.reshape %[[R]] : (tensor<3x8x4x3x1xf64>) -> tensor<3x1x8x4x3xf64>
// CHECK-NEXT: %[[O:[^ ,:)]+]] = stablehlo.reshape %[[V]] : (tensor<3x1x8x4x3xf64>) -> tensor<3x8x4x3xf64>
// CHECK-NEXT: return %[[O]] : tensor<3x8x4x3xf64>
// CHECK-NEXT: }

// Reductions stacked through transpose-like broadcasts (the form
// transpose_like_broadcast_concat leaves on the operands): the broadcast
// reapplied over the stacked reduction.
func.func @stacked_broadcast(%a: tensor<3x8x4x4xf64>, %b: tensor<3x8x4x4xf64>, %c: tensor<3x8x4x4xf64>) -> tensor<3x8x4x4xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %rc = stablehlo.reduce(%c init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %ba = stablehlo.broadcast_in_dim %ra, dims = [1, 3, 2] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
  %bb = stablehlo.broadcast_in_dim %rb, dims = [1, 3, 2] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
  %bc = stablehlo.broadcast_in_dim %rc, dims = [1, 3, 2] : (tensor<8x4x4xf64>) -> tensor<1x8x4x4xf64>
  %r = stablehlo.concatenate %ba, %bb, %bc, dim = 0 : (tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>, tensor<1x8x4x4xf64>) -> tensor<3x8x4x4xf64>
  return %r : tensor<3x8x4x4xf64>
}

// CHECK-LABEL: func.func @stacked_broadcast(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<3x8x4x4xf64>, %[[B:[^ ,:)]+]]: tensor<3x8x4x4xf64>, %[[C:[^ ,:)]+]]: tensor<3x8x4x4xf64>) -> tensor<3x8x4x4xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[UA:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<3x8x4x4xf64>) -> tensor<1x3x8x4x4xf64>
// CHECK-NEXT: %[[UB:[^ ,:)]+]] = stablehlo.reshape %[[B]] : (tensor<3x8x4x4xf64>) -> tensor<1x3x8x4x4xf64>
// CHECK-NEXT: %[[UC:[^ ,:)]+]] = stablehlo.reshape %[[C]] : (tensor<3x8x4x4xf64>) -> tensor<1x3x8x4x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.concatenate %[[UA]], %[[UB]], %[[UC]], dim = 0 : (tensor<1x3x8x4x4xf64>, tensor<1x3x8x4x4xf64>, tensor<1x3x8x4x4xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[S]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[V:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[R]], dims = [0, 2, 4, 3] : (tensor<3x8x4x4xf64>) -> tensor<3x1x8x4x4xf64>
// CHECK-NEXT: %[[O:[^ ,:)]+]] = stablehlo.reshape %[[V]] : (tensor<3x1x8x4x4xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: return %[[O]] : tensor<3x8x4x4xf64>
// CHECK-NEXT: }

// A reduction keeping a unit dimension the other does not have (its
// result reshaped to match): the inputs differ in rank, so no concatenate
// of them exists; left alone (delete_dims_reduce folds that reshape).
func.func @kept_unit(%x: tensor<8x4x4x3xf64>, %y: tensor<8x4x4x3x1xf64>) -> tensor<8x4x6xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x4x4x3xf64>, tensor<f64>) -> tensor<8x4x3xf64>
  %1 = stablehlo.reduce(%y init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x4x4x3x1xf64>, tensor<f64>) -> tensor<8x4x3x1xf64>
  %r = stablehlo.reshape %1 : (tensor<8x4x3x1xf64>) -> tensor<8x4x3xf64>
  %c = stablehlo.concatenate %0, %r, dim = 2 : (tensor<8x4x3xf64>, tensor<8x4x3xf64>) -> tensor<8x4x6xf64>
  return %c : tensor<8x4x6xf64>
}

// CHECK-LABEL: func.func @kept_unit(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<8x4x4x3xf64>, %[[Y:[^ ,:)]+]]: tensor<8x4x4x3x1xf64>) -> tensor<8x4x6xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[RX:[^ ,:)]+]] = stablehlo.reduce(%[[X]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x4x4x3xf64>, tensor<f64>) -> tensor<8x4x3xf64>
// CHECK-NEXT: %[[RY:[^ ,:)]+]] = stablehlo.reduce(%[[Y]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x4x4x3x1xf64>, tensor<f64>) -> tensor<8x4x3x1xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[RY]] : (tensor<8x4x3x1xf64>) -> tensor<8x4x3xf64>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[RX]], %[[R]], dim = 2 : (tensor<8x4x3xf64>, tensor<8x4x3xf64>) -> tensor<8x4x6xf64>
// CHECK-NEXT: return %[[C]] : tensor<8x4x6xf64>
// CHECK-NEXT: }

// A piece reducing its stacked dimension placed elsewhere (dimension 1 of
// [8, 5, 5, 5] beside dimension 0 of [5, 8, 5, 5]): the same reduction of
// its input transposed into the first's form.
func.func @permuted_input(%a: tensor<5x8x5x5xf64>, %b: tensor<8x5x5x5xf64>, %c: tensor<5x8x5x5xf64>) -> tensor<8x5x15xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %z) applies stablehlo.add across dimensions = [0] : (tensor<5x8x5x5xf64>, tensor<f64>) -> tensor<8x5x5xf64>
  %rb = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x5x5x5xf64>, tensor<f64>) -> tensor<8x5x5xf64>
  %rc = stablehlo.reduce(%c init: %z) applies stablehlo.add across dimensions = [0] : (tensor<5x8x5x5xf64>, tensor<f64>) -> tensor<8x5x5xf64>
  %r = stablehlo.concatenate %ra, %rb, %rc, dim = 2 : (tensor<8x5x5xf64>, tensor<8x5x5xf64>, tensor<8x5x5xf64>) -> tensor<8x5x15xf64>
  return %r : tensor<8x5x15xf64>
}

// CHECK-LABEL: func.func @permuted_input(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<5x8x5x5xf64>, %[[B:[^ ,:)]+]]: tensor<8x5x5x5xf64>, %[[C:[^ ,:)]+]]: tensor<5x8x5x5xf64>) -> tensor<8x5x15xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[BT:[^ ,:)]+]] = stablehlo.transpose %[[B]], dims = [1, 0, 2, 3] : (tensor<8x5x5x5xf64>) -> tensor<5x8x5x5xf64>
// CHECK-NEXT: %[[CC:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[BT]], %[[C]], dim = 3 : (tensor<5x8x5x5xf64>, tensor<5x8x5x5xf64>, tensor<5x8x5x5xf64>) -> tensor<5x8x5x15xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[CC]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<5x8x5x15xf64>, tensor<f64>) -> tensor<8x5x15xf64>
// CHECK-NEXT: return %[[R]] : tensor<8x5x15xf64>
// CHECK-NEXT: }
