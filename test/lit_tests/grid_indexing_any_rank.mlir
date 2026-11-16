// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A scatter of single elements into a box of an operand of rank four, with
// a batching dimension, is a dynamic_update_slice of the box.
func.func @scatter_box(%x: tensor<4x8x8x16xf64>, %u: tensor<4x3x3x3xf64>) -> tensor<4x8x8x16xf64> {
  %c = stablehlo.constant dense<[[[[0,0,0],[0,0,1],[0,0,2]],[[0,1,0],[0,1,1],[0,1,2]],[[0,2,0],[0,2,1],[0,2,2]]],[[[1,0,0],[1,0,1],[1,0,2]],[[1,1,0],[1,1,1],[1,1,2]],[[1,2,0],[1,2,1],[1,2,2]]],[[[2,0,0],[2,0,1],[2,0,2]],[[2,1,0],[2,1,1],[2,1,2]],[[2,2,0],[2,2,1],[2,2,2]]]]> : tensor<3x3x3x3xi64>
  %i = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3, 4] : (tensor<3x3x3x3xi64>) -> tensor<4x3x3x3x3xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2, 3], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2, 3], index_vector_dim = 4>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4x8x8x16xf64>, tensor<4x3x3x3x3xi64>, tensor<4x3x3x3xf64>) -> tensor<4x8x8x16xf64>
  return %s : tensor<4x8x8x16xf64>
}

// CHECK-LABEL: func.func @scatter_box(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x8x16xf64>, %[[V1:[^ ,:)]+]]: tensor<4x3x3x3xf64>) -> tensor<4x8x8x16xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 1, 2, 3] : (tensor<4x3x3x3xf64>) -> tensor<4x3x3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V3]], %[[V2]], %[[V2]], %[[V2]], %[[V2]] : (tensor<4x8x8x16xf64>, tensor<4x3x3x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x8x16xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4x8x8x16xf64>
// CHECK-NEXT: }

func.func @scatter_box_no_batching(%x: tensor<8x8x16xf64>, %u: tensor<3x3x3xf64>) -> tensor<8x8x16xf64> {
  %c = stablehlo.constant dense<[[[[0,0,0],[0,0,1],[0,0,2]],[[0,1,0],[0,1,1],[0,1,2]],[[0,2,0],[0,2,1],[0,2,2]]],[[[1,0,0],[1,0,1],[1,0,2]],[[1,1,0],[1,1,1],[1,1,2]],[[1,2,0],[1,2,1],[1,2,2]]],[[[2,0,0],[2,0,1],[2,0,2]],[[2,1,0],[2,1,1],[2,1,2]],[[2,2,0],[2,2,1],[2,2,2]]]]> : tensor<3x3x3x3xi64>
  %s = "stablehlo.scatter"(%x, %c, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x8x16xf64>, tensor<3x3x3x3xi64>, tensor<3x3x3xf64>) -> tensor<8x8x16xf64>
  return %s : tensor<8x8x16xf64>
}

// CHECK-LABEL: func.func @scatter_box_no_batching(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8x16xf64>, %[[V1:[^ ,:)]+]]: tensor<3x3x3xf64>) -> tensor<8x8x16xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 1, 2] : (tensor<3x3x3xf64>) -> tensor<3x3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V3]], %[[V2]], %[[V2]], %[[V2]] : (tensor<8x8x16xf64>, tensor<3x3x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x8x16xf64>
// CHECK-NEXT: return %[[V4]] : tensor<8x8x16xf64>
// CHECK-NEXT: }

// Stays: the second component reaches 9, past its dimension of 8, so the
// scatter drops those elements where the flattened operand would not.
func.func @scatter_past(%x: tensor<8x8xf64>, %u: tensor<2x4xf64>) -> tensor<8x8xf64> {
  %i = stablehlo.constant dense<[[[0,6],[0,7],[0,8],[0,9]],[[1,6],[1,7],[1,8],[1,9]]]> : tensor<2x4x2xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x8xf64>, tensor<2x4x2xi64>, tensor<2x4xf64>) -> tensor<8x8xf64>
  return %s : tensor<8x8xf64>
}

// CHECK-LABEL: func.func @scatter_past(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>, %[[V1:[^ ,:)]+]]: tensor<2x4xf64>) -> tensor<8x8xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[0, 6], [0, 7], [0, 8], [0, 9]], {{\[\[}}1, 6], [1, 7], [1, 8], [1, 9]]]> : tensor<2x4x2xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V2]], %[[V1]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
// CHECK-NEXT: ^bb0(%[[V4:[^ ,:)]+]]: tensor<f64>, %[[V5:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V5]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<8x8xf64>, tensor<2x4x2xi64>, tensor<2x4xf64>) -> tensor<8x8xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x8xf64>
// CHECK-NEXT: }

// A gather of a box is a slice of it.
func.func @gather_box(%x: tensor<4x8x8x16xf64>) -> tensor<4x3x3x3xf64> {
  %c = stablehlo.constant dense<[[[[0,0,0],[0,0,1],[0,0,2]],[[0,1,0],[0,1,1],[0,1,2]],[[0,2,0],[0,2,1],[0,2,2]]],[[[1,0,0],[1,0,1],[1,0,2]],[[1,1,0],[1,1,1],[1,1,2]],[[1,2,0],[1,2,1],[1,2,2]]],[[[2,0,0],[2,0,1],[2,0,2]],[[2,1,0],[2,1,1],[2,1,2]],[[2,2,0],[2,2,1],[2,2,2]]]]> : tensor<3x3x3x3xi64>
  %i = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3, 4] : (tensor<3x3x3x3xi64>) -> tensor<4x3x3x3x3xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2, 3], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2, 3], index_vector_dim = 4>, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<4x8x8x16xf64>, tensor<4x3x3x3x3xi64>) -> tensor<4x3x3x3xf64>
  return %g : tensor<4x3x3x3xf64>
}

// CHECK-LABEL: func.func @gather_box(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x8x16xf64>) -> tensor<4x3x3x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:4, 0:3, 0:3, 0:3] : (tensor<4x8x8x16xf64>) -> tensor<4x3x3x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1, 2, 3] : (tensor<4x3x3x3xf64>) -> tensor<4x3x3x3xf64>
// CHECK-NEXT: return %[[V2]] : tensor<4x3x3x3xf64>
// CHECK-NEXT: }

// The grid runs the box with its first two dimensions swapped.
func.func @gather_transposed(%x: tensor<8x8x16xf64>) -> tensor<3x3x3xf64> {
  %i = stablehlo.constant dense<[[[[0,0,0],[0,0,1],[0,0,2]],[[1,0,0],[1,0,1],[1,0,2]],[[2,0,0],[2,0,1],[2,0,2]]],[[[0,1,0],[0,1,1],[0,1,2]],[[1,1,0],[1,1,1],[1,1,2]],[[2,1,0],[2,1,1],[2,1,2]]],[[[0,2,0],[0,2,1],[0,2,2]],[[1,2,0],[1,2,1],[1,2,2]],[[2,2,0],[2,2,1],[2,2,2]]]]> : tensor<3x3x3x3xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<8x8x16xf64>, tensor<3x3x3x3xi64>) -> tensor<3x3x3xf64>
  return %g : tensor<3x3x3xf64>
}

// CHECK-LABEL: func.func @gather_transposed(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8x16xf64>) -> tensor<3x3x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:3, 0:3, 0:3] : (tensor<8x8x16xf64>) -> tensor<3x3x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [1, 0, 2] : (tensor<3x3x3xf64>) -> tensor<3x3x3xf64>
// CHECK-NEXT: return %[[V2]] : tensor<3x3x3xf64>
// CHECK-NEXT: }

// Stays: the gather clamps each component, not the linear index.
func.func @gather_past(%x: tensor<8x8xf64>) -> tensor<2x4xf64> {
  %i = stablehlo.constant dense<[[[0,6],[0,7],[0,8],[0,9]],[[1,6],[1,7],[1,8],[1,9]]]> : tensor<2x4x2xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8x8xf64>, tensor<2x4x2xi64>) -> tensor<2x4xf64>
  return %g : tensor<2x4xf64>
}

// CHECK-LABEL: func.func @gather_past(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>) -> tensor<2x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[0, 6], [0, 7], [0, 8], [0, 9]], {{\[\[}}1, 6], [1, 7], [1, 8], [1, 9]]]> : tensor<2x4x2xi64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8x8xf64>, tensor<2x4x2xi64>) -> tensor<2x4xf64>
// CHECK-NEXT: return %[[V2]] : tensor<2x4xf64>
// CHECK-NEXT: }

// The diagonal steps 9 elements of the flattened operand, which do not
// divide its 64: a strided slice of it all the same.
func.func @gather_diagonal(%x: tensor<8x8xf64>) -> tensor<3xf64> {
  %i = stablehlo.constant dense<[[0,0],[1,1],[2,2]]> : tensor<3x2xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<8x8xf64>, tensor<3x2xi64>) -> tensor<3xf64>
  return %g : tensor<3xf64>
}

// CHECK-LABEL: func.func @gather_diagonal(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>) -> tensor<3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x8xf64>) -> tensor<64xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:27:9] : (tensor<64xf64>) -> tensor<3xf64>
// CHECK-NEXT: return %[[V2]] : tensor<3xf64>
// CHECK-NEXT: }
