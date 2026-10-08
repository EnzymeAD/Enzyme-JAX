// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_out_of_bounds_noop" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Every index -1: every update dropped, the scatter writes nothing.
func.func @negative(%x: tensor<16xf64>, %u: tensor<4x8xf64>) -> tensor<16xf64> {
  %idx = stablehlo.constant dense<-1> : tensor<4x8x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16xf64>, tensor<4x8x1xi64>, tensor<4x8xf64>) -> tensor<16xf64>
  return %0 : tensor<16xf64>
}

// CHECK:  func.func @negative(%arg0: tensor<16xf64>, %arg1: tensor<4x8xf64>) -> tensor<16xf64> {
// CHECK-NEXT:    return %arg0 : tensor<16xf64>
// CHECK-NEXT:  }

// -----

// Every index past the end of a dimension of window one: dropped too.
func.func @past_end(%x: tensor<16xf64>, %u: tensor<4xf64>) -> tensor<16xf64> {
  %idx = stablehlo.constant dense<16> : tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
  return %0 : tensor<16xf64>
}

// CHECK:  func.func @past_end(%arg0: tensor<16xf64>, %arg1: tensor<4xf64>) -> tensor<16xf64> {
// CHECK-NEXT:    return %arg0 : tensor<16xf64>
// CHECK-NEXT:  }

// -----

// One index in range for all: a write, kept.
func.func @in_range(%x: tensor<16xf64>, %u: tensor<4xf64>) -> tensor<16xf64> {
  %idx = stablehlo.constant dense<3> : tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
  return %0 : tensor<16xf64>
}

// CHECK:  func.func @in_range(%arg0: tensor<16xf64>, %arg1: tensor<4xf64>) -> tensor<16xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<4x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({

// -----

// A window of four rows starting at 14 on a dimension of 16: the start
// is in range, so left alone.
func.func @windowed(%x: tensor<16x8xf64>, %u: tensor<4x4x8xf64>) -> tensor<16x8xf64> {
  %idx = stablehlo.constant dense<14> : tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16x8xf64>, tensor<4x1xi64>, tensor<4x4x8xf64>) -> tensor<16x8xf64>
  return %0 : tensor<16x8xf64>
}

// CHECK:  func.func @windowed(%arg0: tensor<16x8xf64>, %arg1: tensor<4x4x8xf64>) -> tensor<16x8xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<14> : tensor<4x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
