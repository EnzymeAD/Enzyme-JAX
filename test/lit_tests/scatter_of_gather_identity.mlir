// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_of_gather_identity" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// y[idx] = y[idx]: the gather keeps the slot as a window of one that a reshape drops.
func.func @unit_window(%y: tensor<432xf64>, %idx: tensor<16x1xi64>) -> tensor<432xf64> {
  %g = "stablehlo.gather"(%y, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<432xf64>, tensor<16x1xi64>) -> tensor<16x1xf64>
  %r = stablehlo.reshape %g : (tensor<16x1xf64>) -> tensor<16xf64>
  %s = "stablehlo.scatter"(%y, %idx, %r) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<432xf64>, tensor<16x1xi64>, tensor<16xf64>) -> tensor<432xf64>
  return %s : tensor<432xf64>
}

// CHECK:  func.func @unit_window(%arg0: tensor<432xf64>, %arg1: tensor<16x1xi64>) -> tensor<432xf64> {
// CHECK-NEXT:    return %arg0 : tensor<432xf64>
// CHECK-NEXT:  }

// The gather of the scatter's own form, rows of a matrix.
func.func @rows(%y: tensor<8x3xf64>, %idx: tensor<4x1xi64>) -> tensor<8x3xf64> {
  %g = "stablehlo.gather"(%y, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8x3xf64>, tensor<4x1xi64>) -> tensor<4x3xf64>
  %s = "stablehlo.scatter"(%y, %idx, %g) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x3xf64>, tensor<4x1xi64>, tensor<4x3xf64>) -> tensor<8x3xf64>
  return %s : tensor<8x3xf64>
}

// CHECK:  func.func @rows(%arg0: tensor<8x3xf64>, %arg1: tensor<4x1xi64>) -> tensor<8x3xf64> {
// CHECK-NEXT:    return %arg0 : tensor<8x3xf64>
// CHECK-NEXT:  }

// Other slots are read than written: the scatter stays.
func.func @other_slots(%y: tensor<432xf64>, %idx: tensor<16x1xi64>, %idx2: tensor<16x1xi64>) -> tensor<432xf64> {
  %g = "stablehlo.gather"(%y, %idx2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<432xf64>, tensor<16x1xi64>) -> tensor<16x1xf64>
  %r = stablehlo.reshape %g : (tensor<16x1xf64>) -> tensor<16xf64>
  %s = "stablehlo.scatter"(%y, %idx, %r) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<432xf64>, tensor<16x1xi64>, tensor<16xf64>) -> tensor<432xf64>
  return %s : tensor<432xf64>
}

// CHECK:  func.func @other_slots
// CHECK:    stablehlo.scatter

// An add body doubles the slots: the scatter stays.
func.func @add_body(%y: tensor<432xf64>, %idx: tensor<16x1xi64>) -> tensor<432xf64> {
  %g = "stablehlo.gather"(%y, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<432xf64>, tensor<16x1xi64>) -> tensor<16x1xf64>
  %r = stablehlo.reshape %g : (tensor<16x1xf64>) -> tensor<16xf64>
  %s = "stablehlo.scatter"(%y, %idx, %r) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %c = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %c : tensor<f64>
  }) : (tensor<432xf64>, tensor<16x1xi64>, tensor<16xf64>) -> tensor<432xf64>
  return %s : tensor<432xf64>
}

// CHECK:  func.func @add_body
// CHECK:    stablehlo.scatter
