// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_chain_permutation" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Two 3x3 blocks in a buffer of 18. For each distance d of 1 and 2 and each
// row i, the slots (i, i+d) and (i+d, i) of every block are swapped through
// the root, the rows past the block masked to a slot past the end: the four
// scatters transpose every block.
func.func @block_transpose(%x: tensor<18xf64>) -> tensor<18xf64> {
  %c3 = stablehlo.constant dense<3> : tensor<2x3x1xi64>
  %c9 = stablehlo.constant dense<9> : tensor<2x3x1xi64>
  %c1 = stablehlo.constant dense<1> : tensor<2x3x1xi64>
  %c2 = stablehlo.constant dense<2> : tensor<2x3x1xi64>
  %oob = stablehlo.constant dense<18> : tensor<2x3x1xi64>
  %blk = stablehlo.iota dim = 0 : tensor<2x3x1xi64>
  %row = stablehlo.iota dim = 1 : tensor<2x3x1xi64>
  %base = stablehlo.multiply %blk, %c9 : tensor<2x3x1xi64>
  %row3 = stablehlo.multiply %row, %c3 : tensor<2x3x1xi64>
  // d = 1
  %col1 = stablehlo.add %row, %c1 : tensor<2x3x1xi64>
  %in1 = stablehlo.compare LT, %col1, %c3 : (tensor<2x3x1xi64>, tensor<2x3x1xi64>) -> tensor<2x3x1xi1>
  %col13 = stablehlo.multiply %col1, %c3 : tensor<2x3x1xi64>
  %up1 = stablehlo.add %base, %row3 : tensor<2x3x1xi64>
  %i1 = stablehlo.add %up1, %col1 : tensor<2x3x1xi64>
  %lo1 = stablehlo.add %base, %col13 : tensor<2x3x1xi64>
  %j1 = stablehlo.add %lo1, %row : tensor<2x3x1xi64>
  %gj1 = "stablehlo.gather"(%x, %j1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<18xf64>, tensor<2x3x1xi64>) -> tensor<2x3x1xf64>
  %uj1 = stablehlo.reshape %gj1 : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
  %mi1 = stablehlo.select %in1, %i1, %oob : tensor<2x3x1xi1>, tensor<2x3x1xi64>
  %s1 = "stablehlo.scatter"(%x, %mi1, %uj1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<18xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<18xf64>
  %gi1 = "stablehlo.gather"(%x, %i1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<18xf64>, tensor<2x3x1xi64>) -> tensor<2x3x1xf64>
  %ui1 = stablehlo.reshape %gi1 : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
  %mj1 = stablehlo.select %in1, %j1, %oob : tensor<2x3x1xi1>, tensor<2x3x1xi64>
  %s2 = "stablehlo.scatter"(%s1, %mj1, %ui1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<18xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<18xf64>
  // d = 2
  %col2 = stablehlo.add %row, %c2 : tensor<2x3x1xi64>
  %in2 = stablehlo.compare LT, %col2, %c3 : (tensor<2x3x1xi64>, tensor<2x3x1xi64>) -> tensor<2x3x1xi1>
  %col23 = stablehlo.multiply %col2, %c3 : tensor<2x3x1xi64>
  %i2 = stablehlo.add %up1, %col2 : tensor<2x3x1xi64>
  %lo2 = stablehlo.add %base, %col23 : tensor<2x3x1xi64>
  %j2 = stablehlo.add %lo2, %row : tensor<2x3x1xi64>
  %gj2 = "stablehlo.gather"(%x, %j2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<18xf64>, tensor<2x3x1xi64>) -> tensor<2x3x1xf64>
  %uj2 = stablehlo.reshape %gj2 : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
  %mi2 = stablehlo.select %in2, %i2, %oob : tensor<2x3x1xi1>, tensor<2x3x1xi64>
  %s3 = "stablehlo.scatter"(%s2, %mi2, %uj2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<18xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<18xf64>
  %gi2 = "stablehlo.gather"(%x, %i2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<18xf64>, tensor<2x3x1xi64>) -> tensor<2x3x1xf64>
  %ui2 = stablehlo.reshape %gi2 : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
  %mj2 = stablehlo.select %in2, %j2, %oob : tensor<2x3x1xi1>, tensor<2x3x1xi64>
  %s4 = "stablehlo.scatter"(%s3, %mj2, %ui2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<18xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<18xf64>
  return %s4 : tensor<18xf64>
}

// CHECK:  func.func @block_transpose(%arg0: tensor<18xf64>) -> tensor<18xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<18xf64>) -> tensor<2x3x3xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<2x3x3xf64>) -> tensor<18xf64>
// CHECK-NEXT:    return %2 : tensor<18xf64>
// CHECK-NEXT:  }

// -----

// Only the pairs at distance one are swapped: a permutation, but not a
// transpose of any view. Left as it is.
func.func @not_a_transpose(%x: tensor<9xf64>) -> tensor<9xf64> {
  %i = stablehlo.constant dense<[[1], [5]]> : tensor<2x1xi64>
  %j = stablehlo.constant dense<[[3], [7]]> : tensor<2x1xi64>
  %gj = "stablehlo.gather"(%x, %j) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  %s1 = "stablehlo.scatter"(%x, %i, %gj) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<9xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<9xf64>
  %gi = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  %s2 = "stablehlo.scatter"(%s1, %j, %gi) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<9xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<9xf64>
  return %s2 : tensor<9xf64>
}

// CHECK:  func.func @not_a_transpose(%arg0: tensor<9xf64>) -> tensor<9xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}1], [5]]> : tensor<2x1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<{{\[\[}}3], [7]]> : tensor<2x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %1 = "stablehlo.scatter"(%arg0, %c, %0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<9xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<9xf64>
// CHECK-NEXT:    %2 = "stablehlo.gather"(%arg0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %3 = "stablehlo.scatter"(%1, %c_0, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<9xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<9xf64>
// CHECK-NEXT:    return %3 : tensor<9xf64>
// CHECK-NEXT:  }

// -----

// A pair swapped and swapped back, the second swap reading the first's state:
// the identity, so the root.
func.func @undone(%x: tensor<4xf64>) -> tensor<4xf64> {
  %i = stablehlo.constant dense<[[1]]> : tensor<1x1xi64>
  %j = stablehlo.constant dense<[[2]]> : tensor<1x1xi64>
  %gj = "stablehlo.gather"(%x, %j) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<1x1xi64>) -> tensor<1xf64>
  %s1 = "stablehlo.scatter"(%x, %i, %gj) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
  %gi = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<1x1xi64>) -> tensor<1xf64>
  %s2 = "stablehlo.scatter"(%s1, %j, %gi) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
  %gj2 = "stablehlo.gather"(%s2, %j) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<1x1xi64>) -> tensor<1xf64>
  %s3 = "stablehlo.scatter"(%s2, %i, %gj2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
  %gi2 = "stablehlo.gather"(%s2, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<1x1xi64>) -> tensor<1xf64>
  %s4 = "stablehlo.scatter"(%s3, %j, %gi2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
  return %s4 : tensor<4xf64>
}

// CHECK:  func.func @undone(%arg0: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    return %arg0 : tensor<4xf64>
// CHECK-NEXT:  }

// -----

// Two pairs of 3x3 blocks: the first block of each pair is swapped with the
// transpose of the second. The view 2x2x3x3 read with its pairs reversed and
// its blocks transposed.
func.func @pair_swapped_transpose(%x: tensor<36xf64>) -> tensor<36xf64> {
  %i = stablehlo.constant dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8], [18], [19], [20], [21], [22], [23], [24], [25], [26]]> : tensor<18x1xi64>
  %j = stablehlo.constant dense<[[9], [12], [15], [10], [13], [16], [11], [14], [17], [27], [30], [33], [28], [31], [34], [29], [32], [35]]> : tensor<18x1xi64>
  %gj = "stablehlo.gather"(%x, %j) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<36xf64>, tensor<18x1xi64>) -> tensor<18xf64>
  %s1 = "stablehlo.scatter"(%x, %i, %gj) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<36xf64>, tensor<18x1xi64>, tensor<18xf64>) -> tensor<36xf64>
  %gi = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<36xf64>, tensor<18x1xi64>) -> tensor<18xf64>
  %s2 = "stablehlo.scatter"(%s1, %j, %gi) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<36xf64>, tensor<18x1xi64>, tensor<18xf64>) -> tensor<36xf64>
  return %s2 : tensor<36xf64>
}

// CHECK:  func.func @pair_swapped_transpose(%arg0: tensor<36xf64>) -> tensor<36xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<36xf64>) -> tensor<2x2x3x3xf64>
// CHECK-NEXT:    %1 = stablehlo.reverse %0, dims = [1] : tensor<2x2x3x3xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2] : (tensor<2x2x3x3xf64>) -> tensor<2x2x3x3xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<2x2x3x3xf64>) -> tensor<36xf64>
// CHECK-NEXT:    return %3 : tensor<36xf64>
// CHECK-NEXT:  }

// -----

// The ends swapped, then the middle: the buffer reversed.
func.func @reversed(%x: tensor<4xf64>) -> tensor<4xf64> {
  %i = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi64>
  %j = stablehlo.constant dense<[[3], [2]]> : tensor<2x1xi64>
  %gj = "stablehlo.gather"(%x, %j) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  %s1 = "stablehlo.scatter"(%x, %i, %gj) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  %gi = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  %s2 = "stablehlo.scatter"(%s1, %j, %gi) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<4xf64>
  return %s2 : tensor<4xf64>
}

// CHECK:  func.func @reversed(%arg0: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<4xf64>
// CHECK-NEXT:    return %0 : tensor<4xf64>
// CHECK-NEXT:  }
