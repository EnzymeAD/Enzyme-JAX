// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A 2x2 lane grid whose lane is or-ed together from its bits, and its row
// and column taken back by a divide and a remainder: no structure the
// reading follows, but constants and iotas alone, whose values are the box
// [1:3, 1:3] of each element.
func.func @lane_bits(%x: tensor<16x4x4xf64>) -> tensor<16x2x2xf64> {
  %i = stablehlo.iota dim = 0 : tensor<2x2x1xi64>
  %j = stablehlo.iota dim = 1 : tensor<2x2x1xi64>
  %one = stablehlo.constant dense<1> : tensor<2x2x1xi64>
  %hi = stablehlo.shift_left %i, %one : tensor<2x2x1xi64>
  %lane = stablehlo.or %hi, %j : tensor<2x2x1xi64>
  %two = stablehlo.constant dense<2> : tensor<2x2x1xi64>
  %r = stablehlo.divide %lane, %two : tensor<2x2x1xi64>
  %c = stablehlo.remainder %lane, %two : tensor<2x2x1xi64>
  %row = stablehlo.add %r, %one : tensor<2x2x1xi64>
  %col = stablehlo.add %c, %one : tensor<2x2x1xi64>
  %idx = stablehlo.concatenate %row, %col, dim = 2 : (tensor<2x2x1xi64>, tensor<2x2x1xi64>) -> tensor<2x2x2xi64>
  %b = stablehlo.broadcast_in_dim %idx, dims = [1, 2, 3] : (tensor<2x2x2xi64>) -> tensor<16x2x2x2xi64>
  %g = "stablehlo.gather"(%x, %b) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<16x4x4xf64>, tensor<16x2x2x2xi64>) -> tensor<16x2x2xf64>
  return %g : tensor<16x2x2xf64>
}

// CHECK-LABEL: func.func @lane_bits(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x4x4xf64>) -> tensor<16x2x2xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:16, 1:3, 1:3] : (tensor<16x4x4xf64>) -> tensor<16x2x2xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1, 2] : (tensor<16x2x2xf64>) -> tensor<16x2x2xf64>
// CHECK-NEXT: return %[[V2]] : tensor<16x2x2xf64>
// CHECK-NEXT: }

// A computation reading a function argument has no values to read.
func.func @data_index(%x: tensor<16x4x4xf64>, %k: tensor<2x2x1xi64>) -> tensor<16x2x2xf64> {
  %two = stablehlo.constant dense<2> : tensor<2x2x1xi64>
  %row = stablehlo.divide %k, %two : tensor<2x2x1xi64>
  %col = stablehlo.remainder %k, %two : tensor<2x2x1xi64>
  %idx = stablehlo.concatenate %row, %col, dim = 2 : (tensor<2x2x1xi64>, tensor<2x2x1xi64>) -> tensor<2x2x2xi64>
  %b = stablehlo.broadcast_in_dim %idx, dims = [1, 2, 3] : (tensor<2x2x2xi64>) -> tensor<16x2x2x2xi64>
  %g = "stablehlo.gather"(%x, %b) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<16x4x4xf64>, tensor<16x2x2x2xi64>) -> tensor<16x2x2xf64>
  return %g : tensor<16x2x2xf64>
}

// CHECK-LABEL: func.func @data_index(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x4x4xf64>, %[[V1:[^ ,:)]+]]: tensor<2x2x1xi64>) -> tensor<16x2x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<2x2x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.divide %[[V1]], %[[V2]] : tensor<2x2x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.remainder %[[V1]], %[[V2]] : tensor<2x2x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.concatenate %[[V3]], %[[V4]], dim = 2 : (tensor<2x2x1xi64>, tensor<2x2x1xi64>) -> tensor<2x2x2xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [1, 2, 3] : (tensor<2x2x2xi64>) -> tensor<16x2x2x2xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V6]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<16x4x4xf64>, tensor<16x2x2x2xi64>) -> tensor<16x2x2xf64>
// CHECK-NEXT: return %[[V7]] : tensor<16x2x2xf64>
// CHECK-NEXT: }
