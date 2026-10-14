// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The index vector is the column index padded in front with the row: the
// component in the padding is the padding value, so the scatter writes row 1.
func.func @scatter_pad_low(%x: tensor<4x3x8xf64>, %u: tensor<4x2x4xf64>) -> tensor<4x3x8xf64> {
  %c = stablehlo.constant dense<[[[0], [1], [2], [3]], [[4], [5], [6], [7]]]> : tensor<2x4x1xi64>
  %b = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<2x4x1xi64>) -> tensor<4x2x4x1xi64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %i = stablehlo.pad %b, %one, low = [0, 0, 0, 1], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<4x2x4x1xi64>, tensor<i64>) -> tensor<4x2x4x2xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<4x3x8xf64>, tensor<4x2x4x2xi64>, tensor<4x2x4xf64>) -> tensor<4x3x8xf64>
  return %s : tensor<4x3x8xf64>
}

// CHECK-LABEL: func.func @scatter_pad_low(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x3x8xf64>, %[[V1:[^ ,:)]+]]: tensor<4x2x4xf64>) -> tensor<4x3x8xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4x3x8xf64>) -> tensor<4x6x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 1, 2] : (tensor<4x2x4xf64>) -> tensor<4x2x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V4]], %[[V5]], %[[V3]], %[[V2]], %[[V3]] : (tensor<4x6x4xf64>, tensor<4x2x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x6x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V6]] : (tensor<4x6x4xf64>) -> tensor<4x3x8xf64>
// CHECK-NEXT: return %[[V7]] : tensor<4x3x8xf64>
// CHECK-NEXT: }

// Padded behind: the gather reads column 2 of rows 2 to 5.
func.func @gather_pad_high(%x: tensor<8x3xf64>) -> tensor<4xf64> {
  %r = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %two = stablehlo.constant dense<2> : tensor<4x1xi64>
  %rows = stablehlo.add %r, %two : tensor<4x1xi64>
  %col = stablehlo.constant dense<2> : tensor<i64>
  %i = stablehlo.pad %rows, %col, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x1xi64>, tensor<i64>) -> tensor<4x2xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x3xf64>, tensor<4x2xi64>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @gather_pad_high(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x3xf64>) -> tensor<24xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [8:20:3] : (tensor<24xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V2]] : tensor<4xf64>
// CHECK-NEXT: }

// Padded along a dimension of the grid as well: not an index vector of its
// own components, left alone.
func.func @gather_pad_grid(%x: tensor<8x3xf64>) -> tensor<5xf64> {
  %r = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %col = stablehlo.constant dense<2> : tensor<i64>
  %i = stablehlo.pad %r, %col, low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<4x1xi64>, tensor<i64>) -> tensor<5x2xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x3xf64>, tensor<5x2xi64>) -> tensor<5xf64>
  return %g : tensor<5xf64>
}

// CHECK-LABEL: func.func @gather_pad_grid(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3xf64>) -> tensor<5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.pad %[[V2]], %[[V1]], low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<4x1xi64>, tensor<i64>) -> tensor<5x2xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V3]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x3xf64>, tensor<5x2xi64>) -> tensor<5xf64>
// CHECK-NEXT: return %[[V4]] : tensor<5xf64>
// CHECK-NEXT: }
