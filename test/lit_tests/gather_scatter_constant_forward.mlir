// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_scatter_constant_forward" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A scratch written at known positions and read back at known positions: the
// read is a gather of the update.
func.func @permute(%u: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %init = stablehlo.constant dense<0.0> : tensor<2x3xf64>
  %wi = stablehlo.constant dense<[[[0, 0], [0, 2]], [[1, 1], [1, 2]]]> : tensor<2x2x2xi64>
  %s = "stablehlo.scatter"(%init, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<2x3xf64>, tensor<2x2x2xi64>, tensor<2x2xf64>) -> tensor<2x3xf64>
  %ri = stablehlo.constant dense<[[[1, 2], [0, 2]], [[1, 1], [0, 0]]]> : tensor<2x2x2xi64>
  %r = "stablehlo.gather"(%s, %ri) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x3xf64>, tensor<2x2x2xi64>) -> tensor<2x2xf64>
  return %r : tensor<2x2xf64>
}

// CHECK:  func.func @permute(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[[1, 1], [0, 1]], [[1, 0], [0, 0]]]> : tensor<2x2x2xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x2xf64>, tensor<2x2x2xi64>) -> tensor<2x2xf64>
// CHECK-NEXT:    return %0 : tensor<2x2xf64>
// CHECK-NEXT:  }

// -----


// A read of a position no scatter wrote comes from the input.
func.func @unwritten(%u: tensor<2xf64>, %init: tensor<4xf64>) -> tensor<3xf64> {
  %wi = stablehlo.constant dense<[[1], [3]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%init, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  %ri = stablehlo.constant dense<[[3], [0], [1]]> : tensor<3x1xi64>
  %r = "stablehlo.gather"(%s, %ri) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<3x1xi64>) -> tensor<3xf64>
  return %r : tensor<3xf64>
}

// CHECK:  func.func @unwritten(%arg0: tensor<2xf64>, %arg1: tensor<4xf64>) -> tensor<3xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
// CHECK-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[1], [0], [0]]> : tensor<3x1xi64>
// CHECK-NEXT{LITERAL}:    %c_1 = stablehlo.constant dense<[[3], [0], [1]]> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg1, %c_1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:    %2 = stablehlo.select %c, %1, %0 : tensor<3xi1>, tensor<3xf64>
// CHECK-NEXT:    return %2 : tensor<3xf64>
// CHECK-NEXT:  }

// -----


// Two scatters in a chain, the later one overwriting a position of the
// earlier: the read takes the later's update there.
func.func @chain(%u1: tensor<2xf64>, %u2: tensor<1xf64>) -> tensor<2xf64> {
  %init = stablehlo.constant dense<0.0> : tensor<4xf64>
  %wi1 = stablehlo.constant dense<[[1], [2]]> : tensor<2x1xi64>
  %s1 = "stablehlo.scatter"(%init, %wi1, %u1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  %wi2 = stablehlo.constant dense<[[2]]> : tensor<1x1xi64>
  %s2 = "stablehlo.scatter"(%s1, %wi2, %u2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
  %ri = stablehlo.constant dense<[[2], [1]]> : tensor<2x1xi64>
  %r = "stablehlo.gather"(%s2, %ri) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}

// CHECK:  func.func @chain(%arg0: tensor<2xf64>, %arg1: tensor<1xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[false, true]> : tensor<2xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<2x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg1, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<1xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %2 = stablehlo.select %c, %1, %0 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    return %2 : tensor<2xf64>
// CHECK-NEXT:  }

// -----


// The scratch read whole (here returned): the chain becomes the gather of
// its update over the input.
func.func @whole(%u: tensor<2xf64>) -> tensor<4xf64> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
  %wi = stablehlo.constant dense<[[3], [0]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%init, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  return %s : tensor<4xf64>
}

// CHECK: func.func @whole(%arg0: tensor<2xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[true, false, false, true]> : tensor<4xi1>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT{LITERAL}:   %c_0 = stablehlo.constant dense<[[1], [0], [0], [0]]> : tensor<4x1xi64>
// CHECK-NEXT:   %0 = "stablehlo.gather"(%arg0, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK-NEXT:   %1 = stablehlo.select %c, %0, %cst : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:   return %1 : tensor<4xf64>
// CHECK-NEXT: }

// -----

// A lone scatter into a value that is not a constant stays: it updates in
// place, where the select would read the whole of its input.
func.func @whole_in_place(%u: tensor<2xf64>, %init: tensor<4xf64>) -> tensor<4xf64> {
  %wi = stablehlo.constant dense<[[3], [0]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%init, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  return %s : tensor<4xf64>
}

// CHECK: func.func @whole_in_place(%arg0: tensor<2xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT{LITERAL}:   %c = stablehlo.constant dense<[[3], [0]]> : tensor<2x1xi64>
// CHECK-NEXT:   %0 = "stablehlo.scatter"(%arg1, %c, %arg0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %0 : tensor<4xf64>
// CHECK-NEXT: }

// -----


// A scatter writing one position twice: which update wins is not defined,
// so it is left alone.
func.func @twice(%u: tensor<2xf64>, %init: tensor<4xf64>) -> tensor<4xf64> {
  %wi = stablehlo.constant dense<[[1], [1]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%init, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  return %s : tensor<4xf64>
}

// CHECK:  func.func @twice(%arg0: tensor<2xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg1, %c, %arg0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %0 : tensor<4xf64>
// CHECK-NEXT:  }

// -----

// A read of positions the chain never wrote reads its input.
func.func @none_written(%u: tensor<2xf64>, %init: tensor<4xf64>) -> tensor<2xf64> {
  %wi = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%init, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  %ri = stablehlo.constant dense<[[3], [2]]> : tensor<2x1xi64>
  %r = "stablehlo.gather"(%s, %ri) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}

// CHECK:  func.func @none_written(%arg0: tensor<2xf64>, %arg1: tensor<4xf64>) -> tensor<2xf64> {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[3], [2]]> : tensor<2x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:    return %0 : tensor<2xf64>
// CHECK-NEXT:  }
// -----

// The chain goes through a select over a constant mask (what a chain read
// whole becomes, a later store continuing it): where the mask holds, the
// read takes the select's gathered value at the index it gathered.
func.func @through_select(%v: tensor<4xf64>, %u: tensor<1xf64>, %init: tensor<4xf64>) -> tensor<3xf64> {
  %m = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %gi = stablehlo.constant dense<[[3], [2], [0], [0]]> : tensor<4x1xi64>
  %g = "stablehlo.gather"(%v, %gi) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  %sel = stablehlo.select %m, %g, %init : tensor<4xi1>, tensor<4xf64>
  %wi = stablehlo.constant dense<[[2]]> : tensor<1x1xi64>
  %s = "stablehlo.scatter"(%sel, %wi, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
  %ri = stablehlo.constant dense<[[0], [2], [3]]> : tensor<3x1xi64>
  %r = "stablehlo.gather"(%s, %ri) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<3x1xi64>) -> tensor<3xf64>
  return %r : tensor<3xf64>
}

// CHECK:  func.func @through_select(%arg0: tensor<4xf64>, %arg1: tensor<1xf64>, %arg2: tensor<4xf64>) -> tensor<3xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[true, false, false]> : tensor<3xi1>
// CHECK-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[3], [0], [0]]> : tensor<3x1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<[false, true, false]> : tensor<3xi1>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<3x1xi64>
// CHECK-NEXT{LITERAL}:    %c_3 = stablehlo.constant dense<[[0], [2], [3]]> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg2, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg1, %c_2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<1xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:    %2 = stablehlo.select %c_1, %1, %0 : tensor<3xi1>, tensor<3xf64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg0, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:    %4 = stablehlo.select %c, %3, %2 : tensor<3xi1>, tensor<3xf64>
// CHECK-NEXT:    return %4 : tensor<3xf64>
// CHECK-NEXT:  }
