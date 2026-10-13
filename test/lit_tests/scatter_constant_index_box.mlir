// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_masked_index_slice" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A lane grid of 4x4x4 padding 2x2x2 live lanes, its mask folded into the
// constant indices (-1 off the live lanes), batched over 16 elements: cut to
// the live box.
func.func @constant_box(%x: tensor<16x3x16xf64>, %u: tensor<16x4x4x4xf64>) -> tensor<16x3x16xf64> {
  %c = stablehlo.constant dense<[[[[2, 0], [2, 2], [-1, -1], [-1, -1]], [[2, 4], [2, 6], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]], [[[2, 1], [2, 3], [-1, -1], [-1, -1]], [[2, 5], [2, 7], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]], [[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]], [[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]]> : tensor<4x4x4x2xi64>
  %i = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3, 4] : (tensor<4x4x4x2xi64>) -> tensor<16x4x4x4x2xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 4>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<16x3x16xf64>, tensor<16x4x4x4x2xi64>, tensor<16x4x4x4xf64>) -> tensor<16x3x16xf64>
  return %s : tensor<16x3x16xf64>
}

// CHECK-LABEL: func.func @constant_box(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x3x16xf64>, %[[V1:[^ ,:)]+]]: tensor<16x4x4x4xf64>) -> tensor<16x3x16xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<"0x0200000000000000000000000000000002000000000000000200000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000040000000000000002000000000000000600000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000010000000000000002000000000000000300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000050000000000000002000000000000000700000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"> : tensor<4x4x4x2xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [1, 2, 3, 4] : (tensor<4x4x4x2xi64>) -> tensor<16x4x4x4x2xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V3]] [0:16, 0:2, 0:2, 0:2, 0:2] : (tensor<16x4x4x4x2xi64>) -> tensor<16x2x2x2x2xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:16, 0:2, 0:2, 0:2] : (tensor<16x4x4x4xf64>) -> tensor<16x2x2x2xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V4]], %[[V5]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 4>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V7:[^ ,:)]+]]: tensor<f64>, %[[V8:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V8]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<16x3x16xf64>, tensor<16x2x2x2x2xi64>, tensor<16x2x2x2xf64>) -> tensor<16x3x16xf64>
// CHECK-NEXT: return %[[V6]] : tensor<16x3x16xf64>
// CHECK-NEXT: }

// The live lanes are not a box: cut to the box that bounds them, the
// indices kept (their -1 still writes nothing).
func.func @constant_bound(%x: tensor<9xf64>, %u: tensor<3x3xf64>) -> tensor<9xf64> {
  %c = stablehlo.constant dense<[[[0], [1], [-1]], [[3], [-1], [-1]], [[-1], [-1], [-1]]]> : tensor<3x3x1xi64>
  %s = "stablehlo.scatter"(%x, %c, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<9xf64>, tensor<3x3x1xi64>, tensor<3x3xf64>) -> tensor<9xf64>
  return %s : tensor<9xf64>
}

// CHECK-LABEL: func.func @constant_bound(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<9xf64>, %[[V1:[^ ,:)]+]]: tensor<3x3xf64>) -> tensor<9xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[0], [1], [-1]], {{\[\[}}3], [-1], [-1]], {{\[\[}}-1], [-1], [-1]]]> : tensor<3x3x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:2, 0:2, 0:1] : (tensor<3x3x1xi64>) -> tensor<2x2x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:2, 0:2] : (tensor<3x3xf64>) -> tensor<2x2xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V3]], %[[V4]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V6:[^ ,:)]+]]: tensor<f64>, %[[V7:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V7]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<9xf64>, tensor<2x2x1xi64>, tensor<2x2xf64>) -> tensor<9xf64>
// CHECK-NEXT: return %[[V5]] : tensor<9xf64>
// CHECK-NEXT: }

// No index in bounds: the scatter writes nothing.
func.func @constant_none(%x: tensor<9xf64>, %u: tensor<2x2xf64>) -> tensor<9xf64> {
  %c = stablehlo.constant dense<[[[-1], [9]], [[10], [-2]]]> : tensor<2x2x1xi64>
  %s = "stablehlo.scatter"(%x, %c, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<9xf64>, tensor<2x2x1xi64>, tensor<2x2xf64>) -> tensor<9xf64>
  return %s : tensor<9xf64>
}

// CHECK-LABEL: func.func @constant_none(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<9xf64>, %[[V1:[^ ,:)]+]]: tensor<2x2xf64>) -> tensor<9xf64> {
// CHECK-NEXT: return %[[V0]] : tensor<9xf64>
// CHECK-NEXT: }

// Every index in bounds (a constant, not a splat): nothing to cut.
func.func @constant_all_live(%x: tensor<9xf64>, %u: tensor<2x2xf64>) -> tensor<9xf64> {
  %c = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>
  %s = "stablehlo.scatter"(%x, %c, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<9xf64>, tensor<2x2x1xi64>, tensor<2x2xf64>) -> tensor<9xf64>
  return %s : tensor<9xf64>
}

// CHECK-LABEL: func.func @constant_all_live(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<9xf64>, %[[V1:[^ ,:)]+]]: tensor<2x2xf64>) -> tensor<9xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[0], [1]], {{\[\[}}2], [3]]]> : tensor<2x2x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V2]], %[[V1]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V4:[^ ,:)]+]]: tensor<f64>, %[[V5:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V5]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<9xf64>, tensor<2x2x1xi64>, tensor<2x2xf64>) -> tensor<9xf64>
// CHECK-NEXT: return %[[V3]] : tensor<9xf64>
// CHECK-NEXT: }
