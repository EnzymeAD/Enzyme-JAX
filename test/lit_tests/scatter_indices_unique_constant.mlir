// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_indices_are_unique" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Constant indices, every one distinct: unique.
func.func @distinct(%x: tensor<8xf64>, %u: tensor<4xf64>) -> tensor<8xf64> {
  %idx = stablehlo.constant dense<[[5], [1], [7], [2]]> : tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// CHECK:  func.func @distinct(%arg0: tensor<8xf64>, %arg1: tensor<4xf64>) -> tensor<8xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}5], [1], [7], [2{{\]\]}}> : tensor<4x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<8xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<8xf64>
// CHECK-NEXT:    return %0 : tensor<8xf64>
// CHECK-NEXT:  }

// -----

// Two points share an index: not unique.
func.func @repeated(%x: tensor<8xf64>, %u: tensor<4xf64>) -> tensor<8xf64> {
  %idx = stablehlo.constant dense<[[5], [1], [5], [2]]> : tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// CHECK:  func.func @repeated(%arg0: tensor<8xf64>, %arg1: tensor<4xf64>) -> tensor<8xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}5], [1], [5], [2{{\]\]}}> : tensor<4x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({

// -----

// Two-component tuples, the index vector along the leading dimension:
// the tuples (0,1), (0,2), (1,1) are distinct though components repeat.
func.func @tuples(%x: tensor<4x4xf64>, %u: tensor<3xf64>) -> tensor<4x4xf64> {
  %idx = stablehlo.constant dense<[[0, 0, 1], [1, 2, 1]]> : tensor<2x3xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 0>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4x4xf64>, tensor<2x3xi64>, tensor<3xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// CHECK:  func.func @tuples(%arg0: tensor<4x4xf64>, %arg1: tensor<3xf64>) -> tensor<4x4xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0, 0, 1], [1, 2, 1{{\]\]}}> : tensor<2x3xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({

// -----

// Scalar indices (the index vector dimension is the rank): one value each.
func.func @scalar(%x: tensor<8xf64>, %u: tensor<4xf64>) -> tensor<8xf64> {
  %idx = stablehlo.constant dense<[3, 0, 6, 3]> : tensor<4xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<4xi64>, tensor<4xf64>) -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// CHECK:  func.func @scalar(%arg0: tensor<8xf64>, %arg1: tensor<4xf64>) -> tensor<8xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[3, 0, 6, 3]> : tensor<4xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
