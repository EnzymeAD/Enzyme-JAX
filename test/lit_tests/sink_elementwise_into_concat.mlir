// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=sink_elementwise_into_concat" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// An elementwise op over a concatenate and a pad along the same dimension
// becomes a concatenate of the op over the pieces, cut at the union of the
// operands' boundaries: the pad's low rows take a broadcast of the padding
// value, and the pieces the pad's operand does not line up with are slices of
// the pad (slice_pad then folds them).

// CHECK-LABEL: @pad_times_concat
// CHECK: %[[PAD:.+]] = stablehlo.pad %arg4, %arg3, low = [1, 0], high = [0, 0]
// CHECK-DAG: %[[LO:.+]] = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
// CHECK-DAG: %[[M0:.+]] = stablehlo.multiply %[[LO]], %arg0 : tensor<1x4xf32>
// CHECK-DAG: %[[S1:.+]] = stablehlo.slice %[[PAD]] [1:6, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
// CHECK-DAG: %[[M1:.+]] = stablehlo.multiply %[[S1]], %arg1 : tensor<5x4xf32>
// CHECK-DAG: %[[S2:.+]] = stablehlo.slice %[[PAD]] [6:8, 0:4] : (tensor<8x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG: %[[M2:.+]] = stablehlo.multiply %[[S2]], %arg2 : tensor<2x4xf32>
// CHECK: %[[R:.+]] = stablehlo.concatenate %[[M0]], %[[M1]], %[[M2]], dim = 0 : (tensor<1x4xf32>, tensor<5x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
// CHECK: return %[[R]]
func.func @pad_times_concat(%a: tensor<1x4xf32>, %b: tensor<5x4xf32>, %c: tensor<2x4xf32>, %v: tensor<f32>, %p: tensor<7x4xf32>) -> tensor<8x4xf32> {
  %cat = stablehlo.concatenate %a, %b, %c, dim = 0 : (tensor<1x4xf32>, tensor<5x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  %pad = stablehlo.pad %p, %v, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<7x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  %m = stablehlo.multiply %pad, %cat : tensor<8x4xf32>
  return %m : tensor<8x4xf32>
}

// -----

// Any other operand is sliced to the pieces.

// CHECK-LABEL: @concat_plus_value
// CHECK-DAG: %[[S0:.+]] = stablehlo.slice %arg2 [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[A0:.+]] = stablehlo.add %arg0, %[[S0]] : tensor<3x4xf32>
// CHECK-DAG: %[[S1:.+]] = stablehlo.slice %arg2 [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
// CHECK-DAG: %[[A1:.+]] = stablehlo.add %arg1, %[[S1]] : tensor<5x4xf32>
// CHECK: %[[R:.+]] = stablehlo.concatenate %[[A0]], %[[A1]], dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
// CHECK: return %[[R]]
func.func @concat_plus_value(%a: tensor<3x4xf32>, %b: tensor<5x4xf32>, %y: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %cat = stablehlo.concatenate %a, %b, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  %r = stablehlo.add %cat, %y : tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// Concatenates along different dimensions are left alone.

// CHECK-LABEL: @different_dims
// CHECK: stablehlo.concatenate %arg0, %arg1, dim = 0
// CHECK: stablehlo.concatenate %arg2, %arg3, dim = 1
// CHECK: stablehlo.add
// CHECK-NOT: stablehlo.slice
func.func @different_dims(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>, %c: tensor<8x4xf32>, %d: tensor<8x4xf32>) -> tensor<8x8xf32> {
  %x = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
  %y = stablehlo.concatenate %c, %d, dim = 1 : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x8xf32>
  %r = stablehlo.add %x, %y : tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// -----

// A pad with interior padding, or one padding two dimensions, is not a
// concatenate of pieces along one dimension.

// CHECK-LABEL: @pad_two_dims
// CHECK: stablehlo.pad
// CHECK: stablehlo.negate
// CHECK-NOT: stablehlo.concatenate
func.func @pad_two_dims(%p: tensor<6x6xf32>, %v: tensor<f32>) -> tensor<8x8xf32> {
  %pad = stablehlo.pad %p, %v, low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<6x6xf32>, tensor<f32>) -> tensor<8x8xf32>
  %r = stablehlo.negate %pad : tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// -----

// A select over a concatenated predicate sinks too, though stablehlo.select
// carries no Elementwise trait.

// CHECK-LABEL: @select_concat_pred
// CHECK-DAG: %[[A0:.+]] = stablehlo.slice %arg2 [0:3, 0:4]
// CHECK-DAG: %[[B0:.+]] = stablehlo.slice %arg3 [0:3, 0:4]
// CHECK-DAG: %[[S0:.+]] = stablehlo.select %arg0, %[[A0]], %[[B0]] : tensor<3x4xi1>, tensor<3x4xf32>
// CHECK-DAG: %[[A1:.+]] = stablehlo.slice %arg2 [3:8, 0:4]
// CHECK-DAG: %[[B1:.+]] = stablehlo.slice %arg3 [3:8, 0:4]
// CHECK-DAG: %[[S1:.+]] = stablehlo.select %arg1, %[[A1]], %[[B1]] : tensor<5x4xi1>, tensor<5x4xf32>
// CHECK: %[[R:.+]] = stablehlo.concatenate %[[S0]], %[[S1]], dim = 0
// CHECK: return %[[R]]
func.func @select_concat_pred(%p: tensor<3x4xi1>, %q: tensor<5x4xi1>, %a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %m = stablehlo.concatenate %p, %q, dim = 0 : (tensor<3x4xi1>, tensor<5x4xi1>) -> tensor<8x4xi1>
  %r = stablehlo.select %m, %a, %b : tensor<8x4xi1>, tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}
