// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_indices_are_unique" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The raiser's guarded lane store: distinct constant offsets per lane,
// broadcast along the element (batching) dimension, and the lanes the guard
// masks off sent out of bounds. The masked ones write nothing, each element
// is its own batch position, and the offsets are distinct: unique.
func.func @guarded_lanes(%buf: tensor<16x8xf64>, %upd: tensor<16x2x2xf64>, %guard: tensor<16x2x2x1xi1>) -> tensor<16x8xf64> {
  %idx = stablehlo.constant dense<[[[0], [1]], [[4], [5]]]> : tensor<2x2x1xi64>
  %oob = stablehlo.constant dense<-1> : tensor<16x2x2x1xi64>
  %b = stablehlo.broadcast_in_dim %idx, dims = [1, 2, 3] : (tensor<2x2x1xi64>) -> tensor<16x2x2x1xi64>
  %s = stablehlo.select %guard, %b, %oob : tensor<16x2x2x1xi1>, tensor<16x2x2x1xi64>
  %r = "stablehlo.scatter"(%buf, %s, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%old: tensor<f64>, %new: tensor<f64>):
    stablehlo.return %new : tensor<f64>
  }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
  return %r : tensor<16x8xf64>
}

// CHECK:      func.func @guarded_lanes(%arg0: tensor<16x8xf64>, %arg1: tensor<16x2x2xf64>, %arg2: tensor<16x2x2x1xi1>) -> tensor<16x8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{.*}}> : tensor<2x2x1xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<-1> : tensor<16x2x2x1xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<2x2x1xi64>) -> tensor<16x2x2x1xi64>
// CHECK-NEXT:   %1 = stablehlo.select %arg2, %0, %c_0 : tensor<16x2x2x1xi1>, tensor<16x2x2x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%arg0, %1, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
// CHECK-NEXT:   return %2 : tensor<16x8xf64>
// CHECK-NEXT: }

// -----




// The same offsets repeated for two lanes: not unique, with or without the guard.
func.func @repeated_lanes(%buf: tensor<16x8xf64>, %upd: tensor<16x2x2xf64>, %guard: tensor<16x2x2x1xi1>) -> tensor<16x8xf64> {
  %idx = stablehlo.constant dense<[[[0], [1]], [[0], [5]]]> : tensor<2x2x1xi64>
  %oob = stablehlo.constant dense<-1> : tensor<16x2x2x1xi64>
  %b = stablehlo.broadcast_in_dim %idx, dims = [1, 2, 3] : (tensor<2x2x1xi64>) -> tensor<16x2x2x1xi64>
  %s = stablehlo.select %guard, %b, %oob : tensor<16x2x2x1xi1>, tensor<16x2x2x1xi64>
  %r = "stablehlo.scatter"(%buf, %s, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%old: tensor<f64>, %new: tensor<f64>):
    stablehlo.return %new : tensor<f64>
  }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
  return %r : tensor<16x8xf64>
}

// CHECK:      func.func @repeated_lanes(%arg0: tensor<16x8xf64>, %arg1: tensor<16x2x2xf64>, %arg2: tensor<16x2x2x1xi1>) -> tensor<16x8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{.*}}> : tensor<2x2x1xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<-1> : tensor<16x2x2x1xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<2x2x1xi64>) -> tensor<16x2x2x1xi64>
// CHECK-NEXT:   %1 = stablehlo.select %arg2, %0, %c_0 : tensor<16x2x2x1xi1>, tensor<16x2x2x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%arg0, %1, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
// CHECK-NEXT:   return %2 : tensor<16x8xf64>
// CHECK-NEXT: }

// -----




// A select whose other branch is in bounds says nothing: left alone.
func.func @inbounds_select(%buf: tensor<16x8xf64>, %upd: tensor<16x2x2xf64>, %guard: tensor<16x2x2x1xi1>) -> tensor<16x8xf64> {
  %idx = stablehlo.constant dense<[[[0], [1]], [[4], [5]]]> : tensor<2x2x1xi64>
  %alt = stablehlo.constant dense<2> : tensor<16x2x2x1xi64>
  %b = stablehlo.broadcast_in_dim %idx, dims = [1, 2, 3] : (tensor<2x2x1xi64>) -> tensor<16x2x2x1xi64>
  %s = stablehlo.select %guard, %b, %alt : tensor<16x2x2x1xi1>, tensor<16x2x2x1xi64>
  %r = "stablehlo.scatter"(%buf, %s, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%old: tensor<f64>, %new: tensor<f64>):
    stablehlo.return %new : tensor<f64>
  }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
  return %r : tensor<16x8xf64>
}

// CHECK:      func.func @inbounds_select(%arg0: tensor<16x8xf64>, %arg1: tensor<16x2x2xf64>, %arg2: tensor<16x2x2x1xi1>) -> tensor<16x8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{.*}}> : tensor<2x2x1xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<16x2x2x1xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<2x2x1xi64>) -> tensor<16x2x2x1xi64>
// CHECK-NEXT:   %1 = stablehlo.select %arg2, %0, %c_0 : tensor<16x2x2x1xi1>, tensor<16x2x2x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%arg0, %1, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
// CHECK-NEXT:   return %2 : tensor<16x8xf64>
// CHECK-NEXT: }

// -----




// A broadcast that repeats the tuple along a non-batching scatter dimension
// repeats the index: left alone.
func.func @repeat_along_scatter_dim(%buf: tensor<16x8xf64>, %upd: tensor<16x2x2xf64>) -> tensor<16x8xf64> {
  %idx = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi64>
  %b = stablehlo.broadcast_in_dim %idx, dims = [1, 3] : (tensor<2x1xi64>) -> tensor<16x2x2x1xi64>
  %r = "stablehlo.scatter"(%buf, %b, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%old: tensor<f64>, %new: tensor<f64>):
    stablehlo.return %new : tensor<f64>
  }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
  return %r : tensor<16x8xf64>
}

// CHECK:      func.func @repeat_along_scatter_dim(%arg0: tensor<16x8xf64>, %arg1: tensor<16x2x2xf64>) -> tensor<16x8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{.*}}> : tensor<2x1xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %c, dims = [1, 3] : (tensor<2x1xi64>) -> tensor<16x2x2x1xi64>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<16x8xf64>, tensor<16x2x2x1xi64>, tensor<16x2x2xf64>) -> tensor<16x8xf64>
// CHECK-NEXT:   return %1 : tensor<16x8xf64>
// CHECK-NEXT: }
