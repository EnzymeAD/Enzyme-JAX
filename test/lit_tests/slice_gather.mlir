// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_gather" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_gather;gather_op_canon<16>;slice_simplify<16>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=FOLD

// A slice along the gather's batch dimensions is a slice of the indices.
func.func @batch(%x: tensor<100xf64>, %idx: tensor<20x32x1xi64>) -> tensor<20x4xf64> {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %s = stablehlo.slice %g [0:20, 0:4] : (tensor<20x32xf64>) -> tensor<20x4xf64>
  return %s : tensor<20x4xf64>
}

// CHECK:    func.func @batch(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> tensor<20x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:4, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x4x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x4x1xi64>) -> tensor<20x4xf64>
// CHECK-NEXT:    return %1 : tensor<20x4xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @batch(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> tensor<20x4xf64> {
// FOLD-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:4, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x4x1xi64>
// FOLD-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x4x1xi64>) -> tensor<20x4xf64>
// FOLD-NEXT:    return %1 : tensor<20x4xf64>
// FOLD-NEXT:  }

// -----

// A batching pair: the operand is sliced along its paired dimension too.
func.func @batching(%x: tensor<20x100xf64>, %idx: tensor<20x32x1xi64>) -> tensor<8x32xf64> {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %s = stablehlo.slice %g [4:12, 0:32] : (tensor<20x32xf64>) -> tensor<8x32xf64>
  return %s : tensor<8x32xf64>
}

// CHECK:    func.func @batching(%arg0: tensor<20x100xf64>, %arg1: tensor<20x32x1xi64>) -> tensor<8x32xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [4:12, 0:32, 0:1] : (tensor<20x32x1xi64>) -> tensor<8x32x1xi64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [4:12, 0:100] : (tensor<20x100xf64>) -> tensor<8x100xf64>
// CHECK-NEXT:    %2 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x100xf64>, tensor<8x32x1xi64>) -> tensor<8x32xf64>
// CHECK-NEXT:    return %2 : tensor<8x32xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @batching(%arg0: tensor<20x100xf64>, %arg1: tensor<20x32x1xi64>) -> tensor<8x32xf64> {
// FOLD-NEXT:    %0 = stablehlo.slice %arg1 [4:12, 0:32, 0:1] : (tensor<20x32x1xi64>) -> tensor<8x32x1xi64>
// FOLD-NEXT:    %1 = stablehlo.slice %arg0 [4:12, 0:100] : (tensor<20x100xf64>) -> tensor<8x100xf64>
// FOLD-NEXT:    %2 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x100xf64>, tensor<8x32x1xi64>) -> tensor<8x32xf64>
// FOLD-NEXT:    return %2 : tensor<8x32xf64>
// FOLD-NEXT:  }

// -----

// Two slices of one gather: each gets its own, the whole never computed.
func.func @two(%x: tensor<100xf64>, %idx: tensor<20x32x1xi64>) -> (tensor<20x4xf64>, tensor<20x2xf64>) {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %s = stablehlo.slice %g [0:20, 0:4] : (tensor<20x32xf64>) -> tensor<20x4xf64>
  %t = stablehlo.slice %g [0:20, 8:10] : (tensor<20x32xf64>) -> tensor<20x2xf64>
  return %s, %t : tensor<20x4xf64>, tensor<20x2xf64>
}

// CHECK:    func.func @two(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> (tensor<20x4xf64>, tensor<20x2xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:10, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x10x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x10x1xi64>) -> tensor<20x10xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:20, 0:4] : (tensor<20x10xf64>) -> tensor<20x4xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %1 [0:20, 8:10] : (tensor<20x10xf64>) -> tensor<20x2xf64>
// CHECK-NEXT:    return %2, %3 : tensor<20x4xf64>, tensor<20x2xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @two(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> (tensor<20x4xf64>, tensor<20x2xf64>) {
// FOLD-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:10, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x10x1xi64>
// FOLD-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x10x1xi64>) -> tensor<20x10xf64>
// FOLD-NEXT:    %2 = stablehlo.slice %1 [0:20, 0:4] : (tensor<20x10xf64>) -> tensor<20x4xf64>
// FOLD-NEXT:    %3 = stablehlo.slice %1 [0:20, 8:10] : (tensor<20x10xf64>) -> tensor<20x2xf64>
// FOLD-NEXT:    return %2, %3 : tensor<20x4xf64>, tensor<20x2xf64>
// FOLD-NEXT:  }

// -----

// A slice along an offset dimension reads part of each window: left alone.
func.func @offset(%x: tensor<100x8xf64>, %idx: tensor<20x1xi64>) -> tensor<20x4xf64> {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<100x8xf64>, tensor<20x1xi64>) -> tensor<20x8xf64>
  %s = stablehlo.slice %g [0:20, 0:4] : (tensor<20x8xf64>) -> tensor<20x4xf64>
  return %s : tensor<20x4xf64>
}

// CHECK:    func.func @offset(%arg0: tensor<100x8xf64>, %arg1: tensor<20x1xi64>) -> tensor<20x4xf64> {
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<100x8xf64>, tensor<20x1xi64>) -> tensor<20x8xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:20, 0:4] : (tensor<20x8xf64>) -> tensor<20x4xf64>
// CHECK-NEXT:    return %1 : tensor<20x4xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @offset(%arg0: tensor<100x8xf64>, %arg1: tensor<20x1xi64>) -> tensor<20x4xf64> {
// FOLD-NEXT:    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<100x8xf64>, tensor<20x1xi64>) -> tensor<20x8xf64>
// FOLD-NEXT:    %1 = stablehlo.slice %0 [0:20, 0:4] : (tensor<20x8xf64>) -> tensor<20x4xf64>
// FOLD-NEXT:    return %1 : tensor<20x4xf64>
// FOLD-NEXT:  }

// -----

// The two slices make up the whole: gathering them apart gathers as much.
func.func @whole(%x: tensor<100xf64>, %idx: tensor<20x32x1xi64>) -> (tensor<20x16xf64>, tensor<20x16xf64>) {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %a = stablehlo.slice %g [0:20, 0:16] : (tensor<20x32xf64>) -> tensor<20x16xf64>
  %b = stablehlo.slice %g [0:20, 16:32] : (tensor<20x32xf64>) -> tensor<20x16xf64>
  return %a, %b : tensor<20x16xf64>, tensor<20x16xf64>
}

// CHECK:    func.func @whole(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> (tensor<20x16xf64>, tensor<20x16xf64>) {
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:20, 0:16] : (tensor<20x32xf64>) -> tensor<20x16xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [0:20, 16:32] : (tensor<20x32xf64>) -> tensor<20x16xf64>
// CHECK-NEXT:    return %1, %2 : tensor<20x16xf64>, tensor<20x16xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @whole(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> (tensor<20x16xf64>, tensor<20x16xf64>) {
// FOLD-NEXT:    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
// FOLD-NEXT:    %1 = stablehlo.slice %0 [0:20, 0:16] : (tensor<20x32xf64>) -> tensor<20x16xf64>
// FOLD-NEXT:    %2 = stablehlo.slice %0 [0:20, 16:32] : (tensor<20x32xf64>) -> tensor<20x16xf64>
// FOLD-NEXT:    return %1, %2 : tensor<20x16xf64>, tensor<20x16xf64>
// FOLD-NEXT:  }

// -----

// A table read in two pieces that together are the whole: each piece gets a
// gather of its own, from the table's rows sliced to it (a piece may be a
// grid where the whole table is not: here rows 6 r + c and rows 6 r + 5 of
// a vector of 30, the whole crossing its blocks of 6).
func.func @table_pieces(%x: tensor<30xf64>) -> (tensor<5x5xf64>, tensor<5x1xf64>) {
  %t = stablehlo.constant dense<[[[[0], [1], [2], [3], [4]], [[6], [7], [8], [9], [10]], [[12], [13], [14], [15], [16]], [[18], [19], [20], [21], [22]], [[24], [25], [26], [27], [28]]], [[[5], [6], [7], [8], [9]], [[11], [12], [13], [14], [15]], [[17], [18], [19], [20], [21]], [[23], [24], [25], [26], [27]], [[29], [30], [31], [32], [33]]]]> : tensor<2x5x5x1xi64>
  %g = "stablehlo.gather"(%x, %t) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<30xf64>, tensor<2x5x5x1xi64>) -> tensor<2x5x5xf64>
  %tr = stablehlo.transpose %g, dims = [0, 2, 1] : (tensor<2x5x5xf64>) -> tensor<2x5x5xf64>
  %a = stablehlo.slice %tr [0:1, 0:5, 0:5] : (tensor<2x5x5xf64>) -> tensor<1x5x5xf64>
  %a2 = stablehlo.reshape %a : (tensor<1x5x5xf64>) -> tensor<5x5xf64>
  %b = stablehlo.slice %g [1:2, 0:5, 0:1] : (tensor<2x5x5xf64>) -> tensor<1x5x1xf64>
  %b2 = stablehlo.reshape %b : (tensor<1x5x1xf64>) -> tensor<5x1xf64>
  return %a2, %b2 : tensor<5x5xf64>, tensor<5x1xf64>
}

// CHECK:    func.func @table_pieces(%arg0: tensor<30xf64>) -> (tensor<5x5xf64>, tensor<5x1xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}{{\[\[}}0], [1], [2], [3], [4]], {{\[\[}}6], [7], [8], [9], [10]], {{\[\[}}12], [13], [14], [15], [16]], {{\[\[}}18], [19], [20], [21], [22]], {{\[\[}}24], [25], [26], [27], [28]]], {{\[\[}}[5], [6], [7], [8], [9]], {{\[\[}}11], [12], [13], [14], [15]], {{\[\[}}17], [18], [19], [20], [21]], {{\[\[}}23], [24], [25], [26], [27]], {{\[\[}}29], [30], [31], [32], [33]]]]> : tensor<2x5x5x1xi64>
// CHECK-NEXT:    %0 = stablehlo.slice %c [1:2, 0:5, 0:1, 0:1] : (tensor<2x5x5x1xi64>) -> tensor<1x5x1x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<30xf64>, tensor<1x5x1x1xi64>) -> tensor<1x5x1xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %c [0:1, 0:5, 0:5, 0:1] : (tensor<2x5x5x1xi64>) -> tensor<1x5x5x1xi64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg0, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<30xf64>, tensor<1x5x5x1xi64>) -> tensor<1x5x5xf64>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<1x5x5xf64>) -> tensor<1x5x5xf64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<1x5x5xf64>) -> tensor<5x5xf64>
// CHECK-NEXT:    %6 = stablehlo.reshape %1 : (tensor<1x5x1xf64>) -> tensor<5x1xf64>
// CHECK-NEXT:    return %5, %6 : tensor<5x5xf64>, tensor<5x1xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @table_pieces(%arg0: tensor<30xf64>) -> (tensor<5x5xf64>, tensor<5x1xf64>) {
// FOLD-NEXT:    %0 = stablehlo.slice %arg0 [29:30] : (tensor<30xf64>) -> tensor<1xf64>
// FOLD-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<0xf64>
// FOLD-NEXT:    %2 = stablehlo.slice %arg0 [5:30:6] : (tensor<30xf64>) -> tensor<5xf64>
// FOLD-NEXT:    %3 = stablehlo.concatenate %2, %1, dim = 0 : (tensor<5xf64>, tensor<0xf64>) -> tensor<5xf64>
// FOLD-NEXT:    %4 = stablehlo.reshape %3 : (tensor<5xf64>) -> tensor<1x5x1xf64>
// FOLD-NEXT:    %5 = stablehlo.reshape %arg0 : (tensor<30xf64>) -> tensor<5x6xf64>
// FOLD-NEXT:    %6 = stablehlo.slice %5 [0:5, 0:5] : (tensor<5x6xf64>) -> tensor<5x5xf64>
// FOLD-NEXT:    %7 = stablehlo.reshape %6 : (tensor<5x5xf64>) -> tensor<5x1x5xf64>
// FOLD-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 0, 2] : (tensor<5x1x5xf64>) -> tensor<1x5x5x1xf64>
// FOLD-NEXT:    %9 = stablehlo.reshape %8 : (tensor<1x5x5x1xf64>) -> tensor<1x5x5xf64>
// FOLD-NEXT:    %10 = stablehlo.transpose %9, dims = [0, 2, 1] : (tensor<1x5x5xf64>) -> tensor<1x5x5xf64>
// FOLD-NEXT:    %11 = stablehlo.reshape %10 : (tensor<1x5x5xf64>) -> tensor<5x5xf64>
// FOLD-NEXT:    %12 = stablehlo.reshape %4 : (tensor<1x5x1xf64>) -> tensor<5x1xf64>
// FOLD-NEXT:    return %11, %12 : tensor<5x5xf64>, tensor<5x1xf64>
// FOLD-NEXT:  }
