// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=gather_of_scatter_simplify;scatter_of_gather_identity" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A batched scatter inserts the slot dimension; the gather of the same slots
// keeps it as a window of one. The gather reads the updates, reshaped.
func.func @unit_window(%x: tensor<8x375xf64>, %idx: tensor<8x5x1xi64>, %u: tensor<8x5xf64>) -> tensor<8x5x1xf64> {
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x375xf64>, tensor<8x5x1xi64>, tensor<8x5xf64>) -> tensor<8x375xf64>
  %1 = "stablehlo.gather"(%0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x375xf64>, tensor<8x5x1xi64>) -> tensor<8x5x1xf64>
  return %1 : tensor<8x5x1xf64>
}

// CHECK:  func.func @unit_window(%arg0: tensor<8x375xf64>, %arg1: tensor<8x5x1xi64>, %arg2: tensor<8x5xf64>) -> tensor<8x5x1xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg2 : (tensor<8x5xf64>) -> tensor<8x5x1xf64>
// CHECK-NEXT:   return %0 : tensor<8x5x1xf64>
// CHECK-NEXT: }

// The gather reads other batches' slots (no batching dims): not the
// scatter's slots.
func.func @other_batches(%x: tensor<8x375xf64>, %idx: tensor<8x5x1xi64>, %u: tensor<8x5xf64>) -> tensor<8x5x8xf64> {
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x375xf64>, tensor<8x5x1xi64>, tensor<8x5xf64>) -> tensor<8x375xf64>
  %1 = "stablehlo.gather"(%0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x375xf64>, tensor<8x5x1xi64>) -> tensor<8x5x8xf64>
  return %1 : tensor<8x5x8xf64>
}

// CHECK:  func.func @other_batches(%arg0: tensor<8x375xf64>, %arg1: tensor<8x5x1xi64>, %arg2: tensor<8x5xf64>) -> tensor<8x5x8xf64> {
// CHECK-NEXT:   %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8x375xf64>, tensor<8x5x1xi64>, tensor<8x5xf64>) -> tensor<8x375xf64>
// CHECK-NEXT:   %1 = "stablehlo.gather"(%0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x375xf64>, tensor<8x5x1xi64>) -> tensor<8x5x8xf64>
// CHECK-NEXT:   return %1 : tensor<8x5x8xf64>
// CHECK-NEXT: }

// A batched scatter writing back the gather of its own slots, the gather
// keeping the slot dimension as a window of one: the scatter is its input.
func.func @identity_batched(%x: tensor<8x375xf64>, %idx: tensor<8x5x1xi64>) -> tensor<8x375xf64> {
  %0 = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x375xf64>, tensor<8x5x1xi64>) -> tensor<8x5x1xf64>
  %1 = stablehlo.reshape %0 : (tensor<8x5x1xf64>) -> tensor<8x5xf64>
  %2 = "stablehlo.scatter"(%x, %idx, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x375xf64>, tensor<8x5x1xi64>, tensor<8x5xf64>) -> tensor<8x375xf64>
  return %2 : tensor<8x375xf64>
}

// CHECK:  func.func @identity_batched(%arg0: tensor<8x375xf64>, %arg1: tensor<8x5x1xi64>) -> tensor<8x375xf64> {
// CHECK-NEXT:   return %arg0 : tensor<8x375xf64>
// CHECK-NEXT: }
