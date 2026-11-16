// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_gather;transpose_scatter" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A gather's batch dimensions follow its start indices: the transpose moves
// onto them.
func.func @gather(%x: tensor<16xf64>, %i: tensor<3x4x1xi64>) -> tensor<4x3xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<3x4x1xi64>) -> tensor<3x4xf64>
  %t = stablehlo.transpose %g, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
  return %t : tensor<4x3xf64>
}

// CHECK-LABEL: func.func @gather(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xf64>, %[[V1:[^ ,:)]+]]: tensor<3x4x1xi64>) -> tensor<4x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [1, 0, 2] : (tensor<3x4x1xi64>) -> tensor<4x3x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x3x1xi64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x3xf64>
// CHECK-NEXT: }

// A batching pair follows its start indices dimension.
func.func @gather_batching(%x: tensor<5x8xf64>, %i: tensor<5x3x1xi64>) -> tensor<3x5xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<5x8xf64>, tensor<5x3x1xi64>) -> tensor<5x3xf64>
  %t = stablehlo.transpose %g, dims = [1, 0] : (tensor<5x3xf64>) -> tensor<3x5xf64>
  return %t : tensor<3x5xf64>
}

// CHECK-LABEL: func.func @gather_batching(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x8xf64>, %[[V1:[^ ,:)]+]]: tensor<5x3x1xi64>) -> tensor<3x5xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [1, 0, 2] : (tensor<5x3x1xi64>) -> tensor<3x5x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<5x8xf64>, tensor<3x5x1xi64>) -> tensor<3x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3x5xf64>
// CHECK-NEXT: }

// Stays: the transpose moves an offset dimension.
func.func @gather_offset_moves(%x: tensor<16x4xf64>, %i: tensor<3x1xi64>) -> tensor<4x3xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>}> : (tensor<16x4xf64>, tensor<3x1xi64>) -> tensor<3x4xf64>
  %t = stablehlo.transpose %g, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
  return %t : tensor<4x3xf64>
}

// CHECK-LABEL: func.func @gather_offset_moves(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x4xf64>, %[[V1:[^ ,:)]+]]: tensor<3x1xi64>) -> tensor<4x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>}> : (tensor<16x4xf64>, tensor<3x1xi64>) -> tensor<3x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x3xf64>
// CHECK-NEXT: }

// Stays: the gather has another use.
func.func @gather_two_uses(%x: tensor<16xf64>, %i: tensor<3x4x1xi64>) -> (tensor<4x3xf64>, tensor<3x4xf64>) {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<3x4x1xi64>) -> tensor<3x4xf64>
  %t = stablehlo.transpose %g, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
  return %t, %g : tensor<4x3xf64>, tensor<3x4xf64>
}

// CHECK-LABEL: func.func @gather_two_uses(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xf64>, %[[V1:[^ ,:)]+]]: tensor<3x4x1xi64>) -> (tensor<4x3xf64>, tensor<3x4xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<3x4x1xi64>) -> tensor<3x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %[[V3]], %[[V2]] : tensor<4x3xf64>, tensor<3x4xf64>
// CHECK-NEXT: }

// The transpose moves onto the scatter's input, the dimensions it names
// renamed.
func.func @scatter(%x: tensor<8x6xf64>, %i: tensor<4x2xi64>, %u: tensor<4xf64>) -> tensor<6x8xf64> {
  %s = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x6xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<8x6xf64>
  %t = stablehlo.transpose %s, dims = [1, 0] : (tensor<8x6xf64>) -> tensor<6x8xf64>
  return %t : tensor<6x8xf64>
}

// CHECK-LABEL: func.func @scatter(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x6xf64>, %[[V1:[^ ,:)]+]]: tensor<4x2xi64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>) -> tensor<6x8xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V0]], dims = [1, 0] : (tensor<8x6xf64>) -> tensor<6x8xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.scatter"(%[[V3]], %[[V1]], %[[V2]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [1, 0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V5:[^ ,:)]+]]: tensor<f64>, %[[V6:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V6]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<6x8xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<6x8xf64>
// CHECK-NEXT: return %[[V4]] : tensor<6x8xf64>
// CHECK-NEXT: }

// The batching pairs, renamed, are sorted by input dimension again.
func.func @scatter_batching(%x: tensor<2x3x8xf64>, %i: tensor<2x3x4x1xi64>, %u: tensor<2x3x4xf64>) -> tensor<3x2x8xf64> {
  %s = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %c = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %c : tensor<f64>
  }) : (tensor<2x3x8xf64>, tensor<2x3x4x1xi64>, tensor<2x3x4xf64>) -> tensor<2x3x8xf64>
  %t = stablehlo.transpose %s, dims = [1, 0, 2] : (tensor<2x3x8xf64>) -> tensor<3x2x8xf64>
  return %t : tensor<3x2x8xf64>
}

// CHECK-LABEL: func.func @scatter_batching(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x3x8xf64>, %[[V1:[^ ,:)]+]]: tensor<2x3x4x1xi64>, %[[V2:[^ ,:)]+]]: tensor<2x3x4xf64>) -> tensor<3x2x8xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V0]], dims = [1, 0, 2] : (tensor<2x3x8xf64>) -> tensor<3x2x8xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.scatter"(%[[V3]], %[[V1]], %[[V2]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [1, 0], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V5:[^ ,:)]+]]: tensor<f64>, %[[V6:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.add %[[V5]], %[[V6]] : tensor<f64>
// CHECK-NEXT: stablehlo.return %[[V7]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<3x2x8xf64>, tensor<2x3x4x1xi64>, tensor<2x3x4xf64>) -> tensor<3x2x8xf64>
// CHECK-NEXT: return %[[V4]] : tensor<3x2x8xf64>
// CHECK-NEXT: }

// Stays: the transpose reorders the window dimensions the updates hold.
func.func @scatter_windows_reorder(%x: tensor<8x3x4xf64>, %i: tensor<2x1xi64>, %u: tensor<2x3x4xf64>) -> tensor<8x4x3xf64> {
  %s = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x3x4xf64>, tensor<2x1xi64>, tensor<2x3x4xf64>) -> tensor<8x3x4xf64>
  %t = stablehlo.transpose %s, dims = [0, 2, 1] : (tensor<8x3x4xf64>) -> tensor<8x4x3xf64>
  return %t : tensor<8x4x3xf64>
}

// CHECK-LABEL: func.func @scatter_windows_reorder(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<2x1xi64>, %[[V2:[^ ,:)]+]]: tensor<2x3x4xf64>) -> tensor<8x4x3xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V1]], %[[V2]]) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%[[V4:[^ ,:)]+]]: tensor<f64>, %[[V5:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V5]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<8x3x4xf64>, tensor<2x1xi64>, tensor<2x3x4xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [0, 2, 1] : (tensor<8x3x4xf64>) -> tensor<8x4x3xf64>
// CHECK-NEXT: return %[[V6]] : tensor<8x4x3xf64>
// CHECK-NEXT: }
