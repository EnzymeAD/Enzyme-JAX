// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_of_broadcast_batching_dims" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Batching a loop's gather from a buffer the loop does not change gives every
// iteration its copy of the buffer: in each batch the gather reads the same
// tensor, so it reads the buffer.
func.func @per_iteration_copy(%x: tensor<10xf64>, %idx: tensor<4x3x1xi64>) -> tensor<4x3xf64> {
  %b = stablehlo.broadcast_in_dim %x, dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<4x10xf64>, tensor<4x3x1xi64>) -> tensor<4x3xf64>
  return %g : tensor<4x3xf64>
}

// Several broadcast dimensions, a window, and a batching dimension of the
// buffer itself, which stays.
func.func @window(%x: tensor<2x10xf64>, %idx: tensor<2x4x3x1xi64>) -> tensor<2x4x3x2xf64> {
  %b = stablehlo.broadcast_in_dim %x, dims = [0, 2] : (tensor<2x10xf64>) -> tensor<2x4x10xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [3], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 2>}> : (tensor<2x4x10xf64>, tensor<2x4x3x1xi64>) -> tensor<2x4x3x2xf64>
  return %g : tensor<2x4x3x2xf64>
}

// A dimension of the broadcast that the gather indexes, not batches over:
// the gather stays.
func.func @indexed(%x: tensor<10xf64>, %idx: tensor<3x2xi64>) -> tensor<3xf64> {
  %b = stablehlo.broadcast_in_dim %x, dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<4x10xf64>, tensor<3x2xi64>) -> tensor<3xf64>
  return %g : tensor<3xf64>
}

// CHECK-LABEL: func.func @per_iteration_copy(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<10xf64>, %[[V1:[^ ,:)]+]]: tensor<4x3x1xi64>) -> tensor<4x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<10xf64>, tensor<4x3x1xi64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %[[V2]] : tensor<4x3xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @window(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x10xf64>, %[[V1:[^ ,:)]+]]: tensor<2x4x3x1xi64>) -> tensor<2x4x3x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [3], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<2x10xf64>, tensor<2x4x3x1xi64>) -> tensor<2x4x3x2xf64>
// CHECK-NEXT: return %[[V2]] : tensor<2x4x3x2xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @indexed(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<10xf64>, %[[V1:[^ ,:)]+]]: tensor<3x2xi64>) -> tensor<3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V2]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<4x10xf64>, tensor<3x2xi64>) -> tensor<3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3xf64>
// CHECK-NEXT: }
