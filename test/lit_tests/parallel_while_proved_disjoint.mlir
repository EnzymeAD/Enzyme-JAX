// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The loop carries no parallel tag, but every iteration reads and writes the
// buffer at the row of a constant table it selects, and the rows are pairwise
// disjoint, so no iteration can see another's writes.
func.func @disjoint_rows(%y: tensor<12xf64>, %s: tensor<f64>) -> tensor<12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %tbl = stablehlo.constant dense<[[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]> : tensor<3x4xi64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<12xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 4] : (tensor<3x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x4xi64>
    %idx = stablehlo.reshape %row : (tensor<1x4xi64>) -> tensor<4x1xi64>
    %old = "stablehlo.gather"(%acc, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<4x1xi64>) -> tensor<4x1xf64>
    %oldr = stablehlo.reshape %old : (tensor<4x1xf64>) -> tensor<4xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %new = stablehlo.add %oldr, %sb : tensor<4xf64>
    %upd = "stablehlo.scatter"(%acc, %idx, %new) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<12xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<12xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %upd : tensor<i64>, tensor<12xf64>
  }
  return %0#1 : tensor<12xf64>
}

// CHECK:  func.func @disjoint_rows(%arg0: tensor<12xf64>, %arg1: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[\[}}0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11{{\]\]}}> : tensor<3x4xi64>
// CHECK-NEXT:   %0 = stablehlo.reshape %c : (tensor<3x4xi64>) -> tensor<3x1x4xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<3x1x4xi64>) -> tensor<3x4x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<3x4x1xi64>) -> tensor<3x4x1xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<3x4x1xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %6 = stablehlo.add %3, %5 : tensor<3x4xf64>
// CHECK-NEXT:   %7 = "stablehlo.scatter"(%arg0, %1, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<3x4x1xi64>, tensor<3x4xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %7 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// The same loop with a table whose rows share an element: iteration 0 and
// iteration 1 both write element 3, so the loop stays.
func.func @overlapping_rows(%y: tensor<12xf64>, %s: tensor<f64>) -> tensor<12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %tbl = stablehlo.constant dense<[[0, 3, 6, 9], [1, 3, 7, 10], [2, 5, 8, 11]]> : tensor<3x4xi64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<12xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 4] : (tensor<3x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x4xi64>
    %idx = stablehlo.reshape %row : (tensor<1x4xi64>) -> tensor<4x1xi64>
    %old = "stablehlo.gather"(%acc, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<4x1xi64>) -> tensor<4x1xf64>
    %oldr = stablehlo.reshape %old : (tensor<4x1xf64>) -> tensor<4xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %new = stablehlo.add %oldr, %sb : tensor<4xf64>
    %upd = "stablehlo.scatter"(%acc, %idx, %new) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<12xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<12xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %upd : tensor<i64>, tensor<12xf64>
  }
  return %0#1 : tensor<12xf64>
}

// CHECK:  func.func @overlapping_rows(%arg0: tensor<12xf64>, %arg1: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<{{\[\[}}0, 3, 6, 9], [1, 3, 7, 10], [2, 5, 8, 11{{\]\]}}> : tensor<3x4xi64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0) : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %c_2, %iterArg, %c, sizes = [1, 4] : (tensor<3x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x4xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %3 = "stablehlo.gather"(%iterArg_3, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<4x1xi64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %4 = stablehlo.reshape %3 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.add %4, %5 : tensor<4xf64>
// CHECK-NEXT:     %7 = "stablehlo.scatter"(%iterArg_3, %2, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<12xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<12xf64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %8, %7 : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// The rows are disjoint as offsets, but each element is shifted by its own
// loop-invariant amount that is not known here: with a shift of [0, 3, 0],
// element 0 of row 1 lands where element 1 of row 0 lands. A vector base
// does not cancel between two accesses, so the loop is kept.
func.func @vector_base(%acc: tensor<12xf64>, %contrib: tensor<3xf64>, %shift: tensor<3xi64>) -> tensor<12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %tbl = stablehlo.constant dense<[[0, 4, 8], [1, 5, 9]]> : tensor<2x3xi64>
  %0:2 = stablehlo.while(%i = %c0, %a = %acc) : tensor<i64>, tensor<12xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 3] : (tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x3xi64>
    %r = stablehlo.reshape %row : (tensor<1x3xi64>) -> tensor<3xi64>
    %idx1 = stablehlo.add %r, %shift : tensor<3xi64>
    %idx = stablehlo.reshape %idx1 : (tensor<3xi64>) -> tensor<3x1xi64>
    %old = "stablehlo.gather"(%a, %idx) {dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<12xf64>, tensor<3x1xi64>) -> tensor<3xf64>
    %new = stablehlo.add %old, %contrib : tensor<3xf64>
    %upd = "stablehlo.scatter"(%a, %idx, %new) ({
    ^bb0(%x: tensor<f64>, %y: tensor<f64>):
      stablehlo.return %y : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<12xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<12xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %upd : tensor<i64>, tensor<12xf64>
  }
  return %0#1 : tensor<12xf64>
}

// CHECK:  func.func @vector_base(%arg0: tensor<12xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xi64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<{{\[\[}}0, 4, 8], [1, 5, 9{{\]\]}}> : tensor<2x3xi64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0) : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %c_2, %iterArg, %c, sizes = [1, 3] : (tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x3xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x3xi64>) -> tensor<3xi64>
// CHECK-NEXT:     %3 = stablehlo.add %2, %arg2 : tensor<3xi64>
// CHECK-NEXT:     %4 = stablehlo.reshape %3 : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK-NEXT:     %5 = "stablehlo.gather"(%iterArg_3, %4) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:     %6 = stablehlo.add %5, %arg1 : tensor<3xf64>
// CHECK-NEXT:     %7 = "stablehlo.scatter"(%iterArg_3, %4, %6) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<12xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<12xf64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %8, %7 : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// The same shift for every element, given as a scalar and then a constant:
// it cancels between iterations, and the rows stay disjoint. The gather
// clamps an index outside the buffer, so the shift must keep every index
// inside: clamped to [0, 1], the largest is 9 + 1 + 1 = 11.
func.func @scalar_base_plus_constant(%acc: tensor<12xf64>, %contrib: tensor<3xf64>, %s: tensor<i64>) -> tensor<12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<3xi64>
  %tbl = stablehlo.constant dense<[[0, 4, 8], [1, 5, 9]]> : tensor<2x3xi64>
  %sc = stablehlo.clamp %c0, %s, %c1 : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %sb = stablehlo.broadcast_in_dim %sc, dims = [] : (tensor<i64>) -> tensor<3xi64>
  %0:2 = stablehlo.while(%i = %c0, %a = %acc) : tensor<i64>, tensor<12xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 3] : (tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x3xi64>
    %r = stablehlo.reshape %row : (tensor<1x3xi64>) -> tensor<3xi64>
    %idx1 = stablehlo.add %r, %sb : tensor<3xi64>
    %idx2 = stablehlo.add %idx1, %one : tensor<3xi64>
    %idx = stablehlo.reshape %idx2 : (tensor<3xi64>) -> tensor<3x1xi64>
    %old = "stablehlo.gather"(%a, %idx) {dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<12xf64>, tensor<3x1xi64>) -> tensor<3xf64>
    %new = stablehlo.add %old, %contrib : tensor<3xf64>
    %upd = "stablehlo.scatter"(%a, %idx, %new) ({
    ^bb0(%x: tensor<f64>, %y: tensor<f64>):
      stablehlo.return %y : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<12xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<12xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %upd : tensor<i64>, tensor<12xf64>
  }
  return %0#1 : tensor<12xf64>
}

// CHECK:  func.func @scalar_base_plus_constant(%arg0: tensor<12xf64>, %arg1: tensor<3xf64>, %arg2: tensor<i64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<{{\[\[}}0, 4, 8], [1, 5, 9{{\]\]}}> : tensor<2x3xi64>
// CHECK-NEXT:   %0 = stablehlo.clamp %c, %arg2, %c_0 : tensor<i64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:   %2 = stablehlo.reshape %c_2 : (tensor<2x3xi64>) -> tensor<2x1x3xi64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<2x1x3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %5 = stablehlo.add %3, %4 : tensor<2x3xi64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %7 = stablehlo.add %5, %6 : tensor<2x3xi64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<2x3xi64>) -> tensor<2x3x1xi64>
// CHECK-NEXT:   %9 = "stablehlo.gather"(%arg0, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<2x3x1xi64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %11 = stablehlo.add %9, %10 : tensor<2x3xf64>
// CHECK-NEXT:   %12 = "stablehlo.scatter"(%arg0, %8, %11) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %12 : tensor<12xf64>
// CHECK-NEXT: }


// -----

// The same loop with the shift unbounded: a large one clamps indices of the
// two rows to the same element, which one iteration writes and the other
// reads. Kept.
func.func @scalar_base_unbounded(%acc: tensor<12xf64>, %contrib: tensor<3xf64>, %s: tensor<i64>) -> tensor<12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<3xi64>
  %tbl = stablehlo.constant dense<[[0, 4, 8], [1, 5, 9]]> : tensor<2x3xi64>
  %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<i64>) -> tensor<3xi64>
  %0:2 = stablehlo.while(%i = %c0, %a = %acc) : tensor<i64>, tensor<12xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 3] : (tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x3xi64>
    %r = stablehlo.reshape %row : (tensor<1x3xi64>) -> tensor<3xi64>
    %idx1 = stablehlo.add %r, %sb : tensor<3xi64>
    %idx2 = stablehlo.add %idx1, %one : tensor<3xi64>
    %idx = stablehlo.reshape %idx2 : (tensor<3xi64>) -> tensor<3x1xi64>
    %old = "stablehlo.gather"(%a, %idx) {dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<12xf64>, tensor<3x1xi64>) -> tensor<3xf64>
    %new = stablehlo.add %old, %contrib : tensor<3xf64>
    %upd = "stablehlo.scatter"(%a, %idx, %new) ({
    ^bb0(%x: tensor<f64>, %y: tensor<f64>):
      stablehlo.return %y : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<12xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<12xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %upd : tensor<i64>, tensor<12xf64>
  }
  return %0#1 : tensor<12xf64>
}

// CHECK:  func.func @scalar_base_unbounded(%arg0: tensor<12xf64>, %arg1: tensor<3xf64>, %arg2: tensor<i64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<{{\[\[}}0, 4, 8], [1, 5, 9{{\]\]}}> : tensor<2x3xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:   %1:2 = stablehlo.while(%iterArg = %c, %iterArg_4 = %arg0) : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %2 = stablehlo.dynamic_slice %c_3, %iterArg, %c, sizes = [1, 3] : (tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x3xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x3xi64>) -> tensor<3xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %0 : tensor<3xi64>
// CHECK-NEXT:     %5 = stablehlo.add %4, %c_2 : tensor<3xi64>
// CHECK-NEXT:     %6 = stablehlo.reshape %5 : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK-NEXT:     %7 = "stablehlo.gather"(%iterArg_4, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK-NEXT:     %8 = stablehlo.add %7, %arg1 : tensor<3xf64>
// CHECK-NEXT:     %9 = "stablehlo.scatter"(%iterArg_4, %6, %8) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<12xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<12xf64>
// CHECK-NEXT:     %10 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %10, %9 : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1#1 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// Iteration i writes slot i+1, and slot 0 inside a nested loop: every
// iteration writes slot 0, so the iterations are not independent. A chain
// through a nested loop is not proved; the loop is kept.
func.func @nested_write(%x: tensor<4xf64>, %buf: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %b = %buf) : tensor<i64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %ip1 = stablehlo.add %i, %c1 : tensor<i64>
    %b1 = stablehlo.dynamic_update_slice %b, %v, %ip1 : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %1:2 = stablehlo.while(%j = %c0, %bb = %b1) : tensor<i64>, tensor<8xf64>
     cond {
      %cj = stablehlo.compare LT, %j, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cj : tensor<i1>
    } do {
      %bb1 = stablehlo.dynamic_update_slice %bb, %v, %c0 : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %bb1 : tensor<i64>, tensor<8xf64>
    }
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %1#1 : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @nested_write(%arg0: tensor<4xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg1) : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %iterArg_3, %1, %2 : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:     %4:2 = stablehlo.while(%iterArg_4 = %c, %iterArg_5 = %3) : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %6 = stablehlo.compare LT, %iterArg_4, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %6 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_5, %1, %c : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:       %7 = stablehlo.add %iterArg_4, %c_0 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %7, %6 : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %5, %4#1 : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// Iteration i writes slot i of %out from the sum of slots 2i and 2i+1 of %t,
// read in a nested loop. %t is not carried: a loop that hands a buffer back
// unchanged is simplified away first (while_simplify), so nothing writes it,
// its reads are no accesses of a carried buffer, and the nested loop runs
// batched. Batched.
func.func @nested_read(%t: tensor<8xf64>, %out: tensor<4xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<4xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%j = %c0, %acc = %z) : tensor<i64>, tensor<f64>
     cond {
      %cj = stablehlo.compare LT, %j, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cj : tensor<i1>
    } do {
      %ij = stablehlo.multiply %i, %c2 : tensor<i64>
      %idx = stablehlo.add %ij, %j : tensor<i64>
      %r = stablehlo.dynamic_slice %t, %idx, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
      %rs = stablehlo.reshape %r : (tensor<1xf64>) -> tensor<f64>
      %a = stablehlo.add %acc, %rs : tensor<f64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %a : tensor<i64>, tensor<f64>
    }
    %v = stablehlo.reshape %1#1 : (tensor<f64>) -> tensor<1xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %v, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %o1 : tensor<i64>, tensor<4xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @nested_read(%arg0: tensor<8xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %2:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %1) : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %7 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %8 = stablehlo.multiply %0, %7 : tensor<4xi64>
// CHECK-NEXT:     %9 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %10 = stablehlo.add %8, %9 : tensor<4xi64>
// CHECK-NEXT:     %11 = stablehlo.reshape %10 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %12 = "stablehlo.gather"(%arg0, %11) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %13 = stablehlo.reshape %12 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %14 = stablehlo.add %iterArg_3, %13 : tensor<4xf64>
// CHECK-NEXT:     %15 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %15, %14 : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %3 = stablehlo.reshape %2#1 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %4 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %6 = "stablehlo.scatter"(%arg1, %5, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %6 : tensor<4xf64>
// CHECK-NEXT: }

// -----

// A nested loop sums slot 0 of %b, as it came into the iteration (the loop
// handed it back unchanged, and while_simplify made it read the state before
// it), and iteration i writes the sum to slot i+1. The read is one of the
// chain's, in a nested region: slot 0, which no iteration writes. Batched.
func.func @nested_read_written(%buf: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %b = %buf) : tensor<i64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%j = %c0, %acc = %z) : tensor<i64>, tensor<f64>
     cond {
      %cj = stablehlo.compare LT, %j, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cj : tensor<i1>
    } do {
      %r = stablehlo.dynamic_slice %b, %c0, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
      %rs = stablehlo.reshape %r : (tensor<1xf64>) -> tensor<f64>
      %a = stablehlo.add %acc, %rs : tensor<f64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %a : tensor<i64>, tensor<f64>
    }
    %v = stablehlo.reshape %1#1 : (tensor<f64>) -> tensor<1xf64>
    %ip1 = stablehlo.add %i, %c1 : tensor<i64>
    %w = stablehlo.dynamic_update_slice %b, %v, %ip1 : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @nested_read_written(%arg0: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<f64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %9 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %9 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %9 = stablehlo.dynamic_slice %arg0, %c_0, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %11 = stablehlo.add %iterArg_3, %10 : tensor<f64>
// CHECK-NEXT:     %12 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %12, %11 : tensor<i64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %2 = stablehlo.reshape %1#1 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.add %0, %3 : tensor<4xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %6 = stablehlo.clamp %c_0, %4, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %8 = "stablehlo.scatter"(%arg0, %7, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %8 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// Iteration i reads slot 2i+1 and writes slot 2i, both affine in i with the
// same scale: the closed form decides that no two iterations meet, without
// enumerating them.
func.func @affine_disjoint(%y: tensor<64xf64>) -> tensor<64xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c32 = stablehlo.constant dense<32> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %b = %y) : tensor<i64>, tensor<64xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c32 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %w = stablehlo.multiply %i, %c2 : tensor<i64>
    %r = stablehlo.add %w, %c1 : tensor<i64>
    %v = stablehlo.dynamic_slice %b, %r, sizes = [1] : (tensor<64xf64>, tensor<i64>) -> tensor<1xf64>
    %b1 = stablehlo.dynamic_update_slice %b, %v, %w : (tensor<64xf64>, tensor<1xf64>, tensor<i64>) -> tensor<64xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %b1 : tensor<i64>, tensor<64xf64>
  }
  return %0#1 : tensor<64xf64>
}

// CHECK:  func.func @affine_disjoint(%arg0: tensor<64xf64>) -> tensor<64xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<63> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:   %2 = stablehlo.multiply %0, %1 : tensor<32xi64>
// CHECK-NEXT:   %3 = stablehlo.slice %arg0 [1:64] : (tensor<64xf64>) -> tensor<63xf64>
// CHECK-NEXT:   %4 = stablehlo.slice %3 [0:63:2] : (tensor<63xf64>) -> tensor<32xf64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<32xf64>) -> tensor<32x1xf64>
// CHECK-NEXT:   %6 = stablehlo.clamp %c_0, %2, %c : (tensor<i64>, tensor<32xi64>, tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<32xi64>) -> tensor<32x1xi64>
// CHECK-NEXT:   %8 = "stablehlo.scatter"(%arg0, %7, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<64xf64>, tensor<32x1xi64>, tensor<32x1xf64>) -> tensor<64xf64>
// CHECK-NEXT:   return %8 : tensor<64xf64>
// CHECK-NEXT: }


// -----

// Iteration i reads slot i+1, which iteration i+1 writes: the closed form
// finds the windows of two iterations one apart meet. Kept.
func.func @affine_overlap(%y: tensor<33xf64>) -> tensor<33xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c32 = stablehlo.constant dense<32> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %b = %y) : tensor<i64>, tensor<33xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c32 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r = stablehlo.add %i, %c1 : tensor<i64>
    %v = stablehlo.dynamic_slice %b, %r, sizes = [1] : (tensor<33xf64>, tensor<i64>) -> tensor<1xf64>
    %b1 = stablehlo.dynamic_update_slice %b, %v, %i : (tensor<33xf64>, tensor<1xf64>, tensor<i64>) -> tensor<33xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %b1 : tensor<i64>, tensor<33xf64>
  }
  return %0#1 : tensor<33xf64>
}

// CHECK:  func.func @affine_overlap(%arg0: tensor<33xf64>) -> tensor<33xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<32> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<33xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_slice %iterArg_2, %1, sizes = [1] : (tensor<33xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %iterArg_2, %2, %iterArg : (tensor<33xf64>, tensor<1xf64>, tensor<i64>) -> tensor<33xf64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %4, %3 : tensor<i64>, tensor<33xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<33xf64>
// CHECK-NEXT: }


// -----

// A hundred iterations, each writing the row of a table computed from iotas:
// row i is {2i, 2i+1}. The rows are read off the table for as many
// iterations as it has, and no two meet. Batched.
func.func @long_table(%acc: tensor<200xf64>, %contrib: tensor<2xf64>) -> tensor<200xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c100 = stablehlo.constant dense<100> : tensor<i64>
  %two = stablehlo.constant dense<2> : tensor<100x2xi64>
  %rows = stablehlo.iota dim = 0 : tensor<100x2xi64>
  %cols = stablehlo.iota dim = 1 : tensor<100x2xi64>
  %r2 = stablehlo.multiply %rows, %two : tensor<100x2xi64>
  %tbl = stablehlo.add %r2, %cols : tensor<100x2xi64>
  %0:2 = stablehlo.while(%i = %c0, %a = %acc) : tensor<i64>, tensor<200xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c100 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 2] : (tensor<100x2xi64>, tensor<i64>, tensor<i64>) -> tensor<1x2xi64>
    %idx = stablehlo.reshape %row : (tensor<1x2xi64>) -> tensor<2x1xi64>
    %upd = "stablehlo.scatter"(%a, %idx, %contrib) ({
    ^bb0(%x: tensor<f64>, %y: tensor<f64>):
      stablehlo.return %y : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<200xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<200xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %upd : tensor<i64>, tensor<200xf64>
  }
  return %0#1 : tensor<200xf64>
}

// CHECK:  func.func @long_table(%arg0: tensor<200xf64>, %arg1: tensor<2xf64>) -> tensor<200xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<100x2xi64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<100x2xi64>
// CHECK-NEXT:   %1 = stablehlo.iota dim = 1 : tensor<100x2xi64>
// CHECK-NEXT:   %2 = stablehlo.multiply %0, %c : tensor<100x2xi64>
// CHECK-NEXT:   %3 = stablehlo.add %2, %1 : tensor<100x2xi64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<100x2xi64>) -> tensor<100x1x2xi64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<100x1x2xi64>) -> tensor<100x2x1xi64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf64>) -> tensor<100x2xf64>
// CHECK-NEXT:   %7 = "stablehlo.scatter"(%arg0, %5, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<200xf64>, tensor<100x2x1xi64>, tensor<100x2xf64>) -> tensor<200xf64>
// CHECK-NEXT:   return %7 : tensor<200xf64>
// CHECK-NEXT: }

// -----

// Every iteration reads slot 0 and writes slot 2i + 1: the read and the
// writes are at different scales, and decided element by element, no write
// reaches slot 0. Iterations reading the same slot do not conflict. Batched.
func.func @fixed_read(%y: tensor<101xf64>) -> tensor<101xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c50 = stablehlo.constant dense<50> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %b = %y) : tensor<i64>, tensor<101xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c50 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r = stablehlo.dynamic_slice %b, %c0, sizes = [1] : (tensor<101xf64>, tensor<i64>) -> tensor<1xf64>
    %i2 = stablehlo.multiply %i, %c2 : tensor<i64>
    %w = stablehlo.add %i2, %c1 : tensor<i64>
    %b1 = stablehlo.dynamic_update_slice %b, %r, %w : (tensor<101xf64>, tensor<1xf64>, tensor<i64>) -> tensor<101xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %b1 : tensor<i64>, tensor<101xf64>
  }
  return %0#1 : tensor<101xf64>
}

// CHECK:  func.func @fixed_read(%arg0: tensor<101xf64>) -> tensor<101xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<100> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<50xi64>
// CHECK-NEXT:   %1 = stablehlo.dynamic_slice %arg0, %c_0, sizes = [1] : (tensor<101xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<50xi64>
// CHECK-NEXT:   %3 = stablehlo.multiply %0, %2 : tensor<50xi64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<50xi64>
// CHECK-NEXT:   %5 = stablehlo.add %3, %4 : tensor<50xi64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<1xf64>) -> tensor<50x1xf64>
// CHECK-NEXT:   %7 = stablehlo.clamp %c_0, %5, %c : (tensor<i64>, tensor<50xi64>, tensor<i64>) -> tensor<50xi64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<50xi64>) -> tensor<50x1xi64>
// CHECK-NEXT:   %9 = "stablehlo.scatter"(%arg0, %8, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<101xf64>, tensor<50x1xi64>, tensor<50x1xf64>) -> tensor<101xf64>
// CHECK-NEXT:   return %9 : tensor<101xf64>
// CHECK-NEXT: }
