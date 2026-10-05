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
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[\[}}0, 3, 6, 9], [1, 3, 7, 10], [2, 5, 8, 11{{\]\]}}> : tensor<3x4xi64>
// CHECK-NEXT:   %0 = stablehlo.reshape %c : (tensor<3x4xi64>) -> tensor<3x1x4xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<3x1x4xi64>) -> tensor<3x4x1xi64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %4 = "stablehlo.scatter"(%arg0, %1, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     %5 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %5 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<3x4x1xi64>, tensor<3x4xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %4 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// The indices are shifted by a vector the loop does not change, so where
// each iteration lands cannot be told; but the loop only adds into the
// buffer, read-modify-write through the same indices, and nothing else
// touches it: a scatter that adds every iteration's contribution at once
// is the same whatever the shift, collisions included.
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
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[\[}}0, 4, 8], [1, 5, 9{{\]\]}}> : tensor<2x3xi64>
// CHECK-NEXT:   %0 = stablehlo.reshape %c : (tensor<2x3xi64>) -> tensor<2x1x3xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<2x1x3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %3 = stablehlo.add %1, %2 : tensor<2x3xi64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<2x3xi64>) -> tensor<2x3x1xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %6 = "stablehlo.scatter"(%arg0, %4, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     %7 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %7 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %6 : tensor<12xf64>
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
// reads. But the gather and the scatter share their indices, so each slot
// is one the iterations add to, and a read clamped back inside feeds a
// write the scatter drops: a scatter-add over every iteration's indices
// computes the same. Batched.
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
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<{{\[\[}}0, 4, 8], [1, 5, 9{{\]\]}}> : tensor<2x3xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %c_0 : (tensor<2x3xi64>) -> tensor<2x1x3xi64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<2x1x3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %4 = stablehlo.add %2, %3 : tensor<2x3xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<3xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:   %6 = stablehlo.add %4, %5 : tensor<2x3xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<2x3xi64>) -> tensor<2x3x1xi64>
// CHECK-NEXT:   %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %9 = "stablehlo.scatter"(%arg0, %7, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     %10 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %10 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %9 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// A nested loop carries the buffer: its writes are placed by both counters,
// and enumerating them shows the outer iterations write disjoint elements.
func.func @nested_counter(%x: tensor<4x3xf64>, %y: tensor<12xf64>) -> tensor<12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<12xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%j = %c0, %buf = %acc) : tensor<i64>, tensor<12xf64>
     cond {
      %c = stablehlo.compare LT, %j, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %row = stablehlo.multiply %i, %c3 : tensor<i64>
      %idx = stablehlo.add %row, %j : tensor<i64>
      %v = stablehlo.dynamic_slice %x, %i, %j, sizes = [1, 1] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %vs = stablehlo.reshape %v : (tensor<1x1xf64>) -> tensor<1xf64>
      %w = stablehlo.dynamic_update_slice %buf, %vs, %idx : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %w : tensor<i64>, tensor<12xf64>
    }
    %in = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in, %1#1 : tensor<i64>, tensor<12xf64>
  }
  return %0#1 : tensor<12xf64>
}

// CHECK:  func.func @nested_counter(%arg0: tensor<4x3xf64>, %arg1: tensor<12xf64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<11> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %3 = stablehlo.multiply %0, %2 : tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<4xi64>) -> tensor<4x3xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<3xi64>) -> tensor<4x3xi64>
// CHECK-NEXT:   %6 = stablehlo.add %4, %5 : tensor<4x3xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %arg0 : (tensor<4x3xf64>) -> tensor<4x1x3xf64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<4x1x3xf64>) -> tensor<4x1x3x1xf64>
// CHECK-NEXT:   %9 = stablehlo.transpose %8, dims = [0, 2, 1, 3] : (tensor<4x1x3x1xf64>) -> tensor<4x3x1x1xf64>
// CHECK-NEXT:   %10 = stablehlo.reshape %9 : (tensor<4x3x1x1xf64>) -> tensor<4x3x1xf64>
// CHECK-NEXT:   %11 = stablehlo.clamp %c_0, %6, %c : (tensor<i64>, tensor<4x3xi64>, tensor<i64>) -> tensor<4x3xi64>
// CHECK-NEXT:   %12 = stablehlo.reshape %11 : (tensor<4x3xi64>) -> tensor<4x3x1xi64>
// CHECK-NEXT:   %13 = "stablehlo.scatter"(%arg1, %12, %10) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<4x3x1xi64>, tensor<4x3x1xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %13 : tensor<12xf64>
// CHECK-NEXT: }

// -----

// An index reads the counter of the loop around this one: the proof is run
// for each of its values, under which the reads and the writes of different
// iterations never meet.
func.func @enclosing_counter(%x: tensor<8xf64>, %y: tensor<16xf64>) -> tensor<16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%o = %c0, %acc = %y) : tensor<i64>, tensor<16xf64>
   cond {
    %c = stablehlo.compare LT, %o, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%i = %c0, %buf = %acc) : tensor<i64>, tensor<16xf64>
     cond {
      %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %o4 = stablehlo.multiply %o, %c4 : tensor<i64>
      %widx = stablehlo.add %o4, %i : tensor<i64>
      %i4 = stablehlo.multiply %i, %c4 : tensor<i64>
      %ridx = stablehlo.add %i4, %o : tensor<i64>
      %r = stablehlo.dynamic_slice %buf, %ridx, sizes = [1] : (tensor<16xf64>, tensor<i64>) -> tensor<1xf64>
      %v = stablehlo.dynamic_slice %x, %widx, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
      %s = stablehlo.add %r, %v : tensor<1xf64>
      %w = stablehlo.dynamic_update_slice %buf, %s, %widx : (tensor<16xf64>, tensor<1xf64>, tensor<i64>) -> tensor<16xf64>
      %in = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %in, %w : tensor<i64>, tensor<16xf64>
    }
    %on = stablehlo.add %o, %c1 : tensor<i64>
    stablehlo.return %on, %1#1 : tensor<i64>, tensor<16xf64>
  }
  return %0#1 : tensor<16xf64>
}

// CHECK:  func.func @enclosing_counter(%arg0: tensor<8xf64>, %arg1: tensor<16xf64>) -> tensor<16xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<15> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %arg1) : tensor<i64>, tensor<16xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:     %2 = stablehlo.multiply %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:     %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %1 : tensor<4xi64>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %6 = stablehlo.multiply %1, %5 : tensor<4xi64>
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %8 = stablehlo.add %6, %7 : tensor<4xi64>
// CHECK-NEXT:     %9 = stablehlo.reshape %8 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %10 = "stablehlo.gather"(%iterArg_4, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %11 = stablehlo.reshape %4 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %12 = "stablehlo.gather"(%arg0, %11) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %13 = stablehlo.add %10, %12 : tensor<4x1xf64>
// CHECK-NEXT:     %14 = stablehlo.clamp %c_0, %4, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %15 = stablehlo.reshape %14 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %16 = "stablehlo.scatter"(%iterArg_4, %15, %13) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<16xf64>
// CHECK-NEXT:     %17 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %17, %16 : tensor<i64>, tensor<16xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<16xf64>
// CHECK-NEXT: }

// -----

// Every iteration adds its contribution to the first two columns of the
// buffer, rebuilt as a concatenate around its own slices: the slots
// accumulate, so the loop is a reduction over its iterations.
func.func @accumulate_slots(%x: tensor<4x2xf64>, %y: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<2x3xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 2] : (tensor<4x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
    %v = stablehlo.reshape %row : (tensor<1x2xf64>) -> tensor<2x1xf64>
    %old0 = stablehlo.slice %acc [0:2, 0:1] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %new0 = stablehlo.add %old0, %v : tensor<2x1xf64>
    %rest = stablehlo.slice %acc [0:2, 1:3] : (tensor<2x3xf64>) -> tensor<2x2xf64>
    %acc1 = stablehlo.concatenate %new0, %rest, dim = 1 : (tensor<2x1xf64>, tensor<2x2xf64>) -> tensor<2x3xf64>
    %old1 = stablehlo.slice %acc1 [0:2, 1:2] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %new1 = stablehlo.add %v, %old1 : tensor<2x1xf64>
    %tail = stablehlo.slice %acc [0:2, 2:3] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %acc2 = stablehlo.concatenate %new0, %new1, %tail, dim = 1 : (tensor<2x1xf64>, tensor<2x1xf64>, tensor<2x1xf64>) -> tensor<2x3xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %acc2 : tensor<i64>, tensor<2x3xf64>
  }
  return %0#1 : tensor<2x3xf64>
}

// CHECK:  func.func @accumulate_slots(%arg0: tensor<4x2xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<4x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<4x1x2xf64>) -> tensor<4x2x1xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %arg1 [0:2, 0:1] : (tensor<2x3xf64>) -> tensor<2x1xf64>
// CHECK-NEXT:   %3 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x2x1xf64>, tensor<f64>) -> tensor<2x1xf64>
// CHECK-NEXT:   %4 = stablehlo.add %2, %3 : tensor<2x1xf64>
// CHECK-NEXT:   %5 = stablehlo.dynamic_update_slice %arg1, %4, %c_0, %c_0 : (tensor<2x3xf64>, tensor<2x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %6 = stablehlo.slice %5 [0:2, 1:2] : (tensor<2x3xf64>) -> tensor<2x1xf64>
// CHECK-NEXT:   %7 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x2x1xf64>, tensor<f64>) -> tensor<2x1xf64>
// CHECK-NEXT:   %8 = stablehlo.add %6, %7 : tensor<2x1xf64>
// CHECK-NEXT:   %9 = stablehlo.dynamic_update_slice %5, %8, %c_0, %c : (tensor<2x3xf64>, tensor<2x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x3xf64>
// CHECK-NEXT:   return %9 : tensor<2x3xf64>
// CHECK-NEXT: }

// -----

// The whole carried buffer is added to every iteration: a reduction over
// the iterations.
func.func @accumulate_whole(%x: tensor<4x3xf64>, %y: tensor<3xf64>) -> tensor<3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<3xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
    %v = stablehlo.reshape %row : (tensor<1x3xf64>) -> tensor<3xf64>
    %s = stablehlo.add %acc, %v : tensor<3xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<3xf64>
  }
  return %0#1 : tensor<3xf64>
}

// CHECK:  func.func @accumulate_whole(%arg0: tensor<4x3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<4x3xf64>) -> tensor<4x1x3xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<4x1x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:   %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT:   %3 = stablehlo.add %arg1, %2 : tensor<3xf64>
// CHECK-NEXT:   return %3 : tensor<3xf64>
// CHECK-NEXT: }

// -----

// Every iteration adds into elements it chooses from a table, and two
// iterations may choose the same element: the batched scatter adds them all.
func.func @scatter_accumulate(%x: tensor<4xf64>, %y: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %tab = stablehlo.constant dense<[[2], [5], [2], [7]]> : tensor<4x1xi64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %idx = stablehlo.dynamic_slice %tab, %i, %c0, sizes = [1, 1] : (tensor<4x1xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
    %old = "stablehlo.gather"(%acc, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<1x1xi64>) -> tensor<1xf64>
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %new = stablehlo.add %old, %v : tensor<1xf64>
    %w = "stablehlo.scatter"(%acc, %idx, %new) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<8xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @scatter_accumulate(%arg0: tensor<4xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[\[}}2], [5], [2], [7{{\]\]}}> : tensor<4x1xi64>
// CHECK-NEXT:   %0 = stablehlo.reshape %c : (tensor<4x1xi64>) -> tensor<4x1x1xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%arg1, %0, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     %3 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8xf64>, tensor<4x1x1xi64>, tensor<4x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %2 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// The buffer rebuilt as a concatenate whose second piece is a concatenate
// along the other dimension of two accumulating slots.
func.func @nested_accumulate(%x: tensor<4x3xf64>, %z: tensor<4x3xf64>, %y: tensor<2x6xf64>) -> tensor<2x6xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<2x6xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %top = stablehlo.slice %acc [0:1, 0:6] : (tensor<2x6xf64>) -> tensor<1x6xf64>
    %l = stablehlo.slice %acc [1:2, 0:3] : (tensor<2x6xf64>) -> tensor<1x3xf64>
    %r = stablehlo.slice %acc [1:2, 3:6] : (tensor<2x6xf64>) -> tensor<1x3xf64>
    %xi = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
    %zi = stablehlo.dynamic_slice %z, %i, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
    %la = stablehlo.add %l, %xi : tensor<1x3xf64>
    %ra = stablehlo.add %r, %zi : tensor<1x3xf64>
    %bot = stablehlo.concatenate %la, %ra, dim = 1 : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<1x6xf64>
    %u = stablehlo.concatenate %top, %bot, dim = 0 : (tensor<1x6xf64>, tensor<1x6xf64>) -> tensor<2x6xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<2x6xf64>
  }
  return %0#1 : tensor<2x6xf64>
}

// CHECK:  func.func @nested_accumulate(%arg0: tensor<4x3xf64>, %arg1: tensor<4x3xf64>, %arg2: tensor<2x6xf64>) -> tensor<2x6xf64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:  %0 = stablehlo.reshape %arg0 : (tensor<4x3xf64>) -> tensor<4x1x3xf64>
// CHECK-NEXT:  %1 = stablehlo.reshape %arg1 : (tensor<4x3xf64>) -> tensor<4x1x3xf64>
// CHECK-NEXT:  %2 = stablehlo.slice %arg2 [1:2, 0:3] : (tensor<2x6xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %3 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x1x3xf64>, tensor<f64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %4 = stablehlo.add %2, %3 : tensor<1x3xf64>
// CHECK-NEXT:  %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_1, %c_0 : (tensor<2x6xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<2x6xf64>
// CHECK-NEXT:  %6 = stablehlo.slice %5 [1:2, 3:6] : (tensor<2x6xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %7 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x1x3xf64>, tensor<f64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %8 = stablehlo.add %6, %7 : tensor<1x3xf64>
// CHECK-NEXT:  %9 = stablehlo.dynamic_update_slice %5, %8, %c_1, %c : (tensor<2x6xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<2x6xf64>
// CHECK-NEXT:  return %9 : tensor<2x6xf64>
// CHECK-NEXT:  }

// -----

// Every iteration writes its own slot and reads the next one, which a later
// iteration writes: sequentially the read sees the value the loop was entered
// with, so batched it is taken from the entry buffer.
func.func @reads_later_slot(%y: tensor<8xf64>, %x: tensor<4xf64>, %out: tensor<4xf64>) -> (tensor<8xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %y, %o = %out) : tensor<i64>, tensor<8xf64>, tensor<4xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %w = stablehlo.dynamic_update_slice %acc, %v, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    %r = stablehlo.dynamic_slice %w, %n, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %o, %r, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    stablehlo.return %n, %w, %u : tensor<i64>, tensor<8xf64>, tensor<4xf64>
  }
  return %0#1, %0#2 : tensor<8xf64>, tensor<4xf64>
}

// CHECK:  func.func @reads_later_slot(%arg0: tensor<8xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<8xf64>, tensor<4xf64>) {
// CHECK-NEXT:  %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %0 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:  %1 = stablehlo.reshape %0 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:  %2 = stablehlo.dynamic_update_slice %arg0, %1, %c : (tensor<8xf64>, tensor<4xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:  %3 = stablehlo.slice %arg0 [1:5] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT:  %4 = stablehlo.reshape %3 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:  %5 = stablehlo.reshape %4 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:  %6 = stablehlo.dynamic_update_slice %arg2, %5, %c : (tensor<4xf64>, tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:  return %2, %6 : tensor<8xf64>, tensor<4xf64>
// CHECK-NEXT:  }

// -----

// The slot read here was written by the previous iteration (and by this one
// for the first, whose index clamps to its own), before the read in the
// chain: batched, the read of the chain after that write sees it too.
func.func @reads_earlier_slot(%y: tensor<8xf64>, %x: tensor<4xf64>, %out: tensor<4xf64>) -> (tensor<8xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %y, %o = %out) : tensor<i64>, tensor<8xf64>, tensor<4xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %w = stablehlo.dynamic_update_slice %acc, %v, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %p = stablehlo.subtract %i, %c1 : tensor<i64>
    %r = stablehlo.dynamic_slice %w, %p, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %o, %r, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w, %u : tensor<i64>, tensor<8xf64>, tensor<4xf64>
  }
  return %0#1, %0#2 : tensor<8xf64>, tensor<4xf64>
}

// CHECK:  func.func @reads_earlier_slot
// CHECK-NOT: stablehlo.while

// -----

// The slot of the previous iteration is read before, in the chain, that
// iteration writes it: sequentially the read sees the write, batched it
// would not. The loop stays.
func.func @reads_before_later_write(%y: tensor<8xf64>, %x: tensor<4xf64>, %out: tensor<4xf64>) -> (tensor<8xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %y, %o = %out) : tensor<i64>, tensor<8xf64>, tensor<4xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %hi = stablehlo.add %i, %c4 : tensor<i64>
    %w = stablehlo.dynamic_update_slice %acc, %v, %hi : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %p = stablehlo.subtract %i, %c1 : tensor<i64>
    %r = stablehlo.dynamic_slice %w, %p, sizes = [5] : (tensor<8xf64>, tensor<i64>) -> tensor<5xf64>
    %r0 = stablehlo.slice %r [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %o, %r0, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %w2 = stablehlo.dynamic_update_slice %w, %v, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w2, %u : tensor<i64>, tensor<8xf64>, tensor<4xf64>
  }
  return %0#1, %0#2 : tensor<8xf64>, tensor<4xf64>
}

// CHECK:  func.func @reads_before_later_write
// CHECK:    stablehlo.while

// -----

// The rows of the index table are computed outside the loop from an iota and
// constants: each iteration's row is evaluated like a constant's.
func.func @computed_table(%y: tensor<24xf64>, %x: tensor<4x2xf64>) -> tensor<24xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<4x2xi64>
  %off = stablehlo.constant dense<[[0], [4]]> : tensor<2x1xi64>
  %lane = stablehlo.iota dim = 0 : tensor<4x2xi64>
  %b = stablehlo.reshape %off : (tensor<2x1xi64>) -> tensor<2xi64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<2xi64>) -> tensor<4x2xi64>
  %t0 = stablehlo.add %lane, %bb : tensor<4x2xi64>
  %tbl = stablehlo.multiply %t0, %c3 : tensor<4x2xi64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<24xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 2] : (tensor<4x2xi64>, tensor<i64>, tensor<i64>) -> tensor<1x2xi64>
    %idx = stablehlo.reshape %row : (tensor<1x2xi64>) -> tensor<2x1xi64>
    %v = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 2] : (tensor<4x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
    %u = stablehlo.reshape %v : (tensor<1x2xf64>) -> tensor<2xf64>
    %s = "stablehlo.scatter"(%acc, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %bv: tensor<f64>):
      stablehlo.return %bv : tensor<f64>
    }) : (tensor<24xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<24xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<24xf64>
  }
  return %0#1 : tensor<24xf64>
}

// CHECK:  func.func @computed_table(%arg0: tensor<24xf64>, %arg1: tensor<4x2xf64>) -> tensor<24xf64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<3> : tensor<4x2xi64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<{{\[\[}}0], [4{{\]\]}}> : tensor<2x1xi64>
// CHECK-NEXT:  %0 = stablehlo.iota dim = 0 : tensor<4x2xi64>
// CHECK-NEXT:  %1 = stablehlo.reshape %c_0 : (tensor<2x1xi64>) -> tensor<2xi64>
// CHECK-NEXT:  %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<2xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:  %3 = stablehlo.add %0, %2 : tensor<4x2xi64>
// CHECK-NEXT:  %4 = stablehlo.multiply %3, %c : tensor<4x2xi64>
// CHECK-NEXT:  %5 = stablehlo.reshape %4 : (tensor<4x2xi64>) -> tensor<4x1x2xi64>
// CHECK-NEXT:  %6 = stablehlo.reshape %5 : (tensor<4x1x2xi64>) -> tensor<4x2x1xi64>
// CHECK-NEXT:  %7 = stablehlo.reshape %arg1 : (tensor<4x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:  %8 = stablehlo.reshape %7 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:  %9 = "stablehlo.scatter"(%arg0, %6, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:  stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<24xf64>, tensor<4x2x1xi64>, tensor<4x2xf64>) -> tensor<24xf64>
// CHECK-NEXT:  return %9 : tensor<24xf64>
// CHECK-NEXT:  }

// -----

// Slots clamped from below with a maximum and spread by an elementwise
// division: the evaluator follows both.
func.func @max_div_index(%y: tensor<16xf64>, %x: tensor<4x2xf64>) -> tensor<16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %tbl = stablehlo.constant dense<[[-2, 1], [2, 5], [6, 9], [10, 13]]> : tensor<4x2xi64>
  %div = stablehlo.constant dense<[[1], [1]]> : tensor<2x1xi64>
  %zero = stablehlo.constant dense<0> : tensor<2x1xi64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<16xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %tbl, %i, %c0, sizes = [1, 2] : (tensor<4x2xi64>, tensor<i64>, tensor<i64>) -> tensor<1x2xi64>
    %r = stablehlo.reshape %row : (tensor<1x2xi64>) -> tensor<2x1xi64>
    %m = stablehlo.maximum %r, %zero : tensor<2x1xi64>
    %idx = stablehlo.divide %m, %div : tensor<2x1xi64>
    %v = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 2] : (tensor<4x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
    %u = stablehlo.reshape %v : (tensor<1x2xf64>) -> tensor<2xf64>
    %s = "stablehlo.scatter"(%acc, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %bv: tensor<f64>):
      stablehlo.return %bv : tensor<f64>
    }) : (tensor<16xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<16xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<16xf64>
  }
  return %0#1 : tensor<16xf64>
}

// CHECK:  func.func @max_div_index(%arg0: tensor<16xf64>, %arg1: tensor<4x2xf64>) -> tensor<16xf64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<{{\[\[}}-2, 1], [2, 5], [6, 9], [10, 13{{\]\]}}> : tensor<4x2xi64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<1> : tensor<2x1xi64>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<0> : tensor<2x1xi64>
// CHECK-NEXT:  %0 = stablehlo.reshape %c : (tensor<4x2xi64>) -> tensor<4x1x2xi64>
// CHECK-NEXT:  %1 = stablehlo.reshape %0 : (tensor<4x1x2xi64>) -> tensor<4x2x1xi64>
// CHECK-NEXT:  %2 = stablehlo.broadcast_in_dim %c_1, dims = [1, 2] : (tensor<2x1xi64>) -> tensor<4x2x1xi64>
// CHECK-NEXT:  %3 = stablehlo.maximum %1, %2 : tensor<4x2x1xi64>
// CHECK-NEXT:  %4 = stablehlo.broadcast_in_dim %c_0, dims = [1, 2] : (tensor<2x1xi64>) -> tensor<4x2x1xi64>
// CHECK-NEXT:  %5 = stablehlo.divide %3, %4 : tensor<4x2x1xi64>
// CHECK-NEXT:  %6 = stablehlo.reshape %arg1 : (tensor<4x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:  %7 = stablehlo.reshape %6 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:  %8 = "stablehlo.scatter"(%arg0, %5, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:  stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<16xf64>, tensor<4x2x1xi64>, tensor<4x2xf64>) -> tensor<16xf64>
// CHECK-NEXT:  return %8 : tensor<16xf64>
// CHECK-NEXT:  }

// -----

// A grid-stride loop unrolled past its certain trips: block b writes element
// b, then element b + 27 under the condition that it exists, the chain
// passing through a select. Batched, the second scatter sends the rows of the
// blocks without a second element out of the buffer.
func.func @guarded_link(%x: tensor<53xf64>, %y: tensor<53xf64>) -> tensor<53xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c26 = stablehlo.constant dense<26> : tensor<i64>
  %c27 = stablehlo.constant dense<27> : tensor<i64>
  %c53 = stablehlo.constant dense<53> : tensor<i64>
  %0:2 = stablehlo.while(%b = %c0, %acc = %y) : tensor<i64>, tensor<53xf64>
   cond {
    %c = stablehlo.compare LT, %b, %c27 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %n = stablehlo.subtract %c53, %b : tensor<i64>
    %m = stablehlo.add %n, %c26 : tensor<i64>
    %trip = stablehlo.divide %m, %c27 : tensor<i64>
    %p = stablehlo.compare LT, %c1, %trip : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %i1 = stablehlo.reshape %b : (tensor<i64>) -> tensor<1x1xi64>
    %v1 = stablehlo.dynamic_slice %x, %b, sizes = [1] : (tensor<53xf64>, tensor<i64>) -> tensor<1xf64>
    %s1 = "stablehlo.scatter"(%acc, %i1, %v1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %bv: tensor<f64>):
      stablehlo.return %bv : tensor<f64>
    }) : (tensor<53xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<53xf64>
    %e2 = stablehlo.add %b, %c27 : tensor<i64>
    %i2 = stablehlo.reshape %e2 : (tensor<i64>) -> tensor<1x1xi64>
    %v2 = stablehlo.dynamic_slice %x, %e2, sizes = [1] : (tensor<53xf64>, tensor<i64>) -> tensor<1xf64>
    %s2 = "stablehlo.scatter"(%s1, %i2, %v2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %bv: tensor<f64>):
      stablehlo.return %bv : tensor<f64>
    }) : (tensor<53xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<53xf64>
    %u = stablehlo.select %p, %s2, %s1 : tensor<i1>, tensor<53xf64>
    %bn = stablehlo.add %b, %c1 : tensor<i64>
    stablehlo.return %bn, %u : tensor<i64>, tensor<53xf64>
  }
  return %0#1 : tensor<53xf64>
}

// CHECK:  func.func @guarded_link(%arg0: tensor<53xf64>, %arg1: tensor<53xf64>) -> tensor<53xf64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<53> : tensor<27x1x1xi64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<26> : tensor<i64>
// CHECK-NEXT:  %c_2 = stablehlo.constant dense<27> : tensor<i64>
// CHECK-NEXT:  %c_3 = stablehlo.constant dense<53> : tensor<i64>
// CHECK-NEXT:  %0 = stablehlo.iota dim = 0 : tensor<27xi64>
// CHECK-NEXT:  %1 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<27xi64>
// CHECK-NEXT:  %2 = stablehlo.subtract %1, %0 : tensor<27xi64>
// CHECK-NEXT:  %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<27xi64>
// CHECK-NEXT:  %4 = stablehlo.add %2, %3 : tensor<27xi64>
// CHECK-NEXT:  %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<27xi64>
// CHECK-NEXT:  %6 = stablehlo.divide %4, %5 : tensor<27xi64>
// CHECK-NEXT:  %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<27xi64>
// CHECK-NEXT:  %8 = stablehlo.compare LT, %7, %6 : (tensor<27xi64>, tensor<27xi64>) -> tensor<27xi1>
// CHECK-NEXT:  %9 = stablehlo.reshape %0 : (tensor<27xi64>) -> tensor<27x1x1xi64>
// CHECK-NEXT:  %10 = stablehlo.slice %arg0 [0:27] : (tensor<53xf64>) -> tensor<27xf64>
// CHECK-NEXT:  %11 = stablehlo.reshape %10 : (tensor<27xf64>) -> tensor<27x1xf64>
// CHECK-NEXT:  %12 = "stablehlo.scatter"(%arg1, %9, %11) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:  stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<53xf64>, tensor<27x1x1xi64>, tensor<27x1xf64>) -> tensor<53xf64>
// CHECK-NEXT:  %13 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<27xi64>
// CHECK-NEXT:  %14 = stablehlo.add %0, %13 : tensor<27xi64>
// CHECK-NEXT:  %15 = stablehlo.reshape %14 : (tensor<27xi64>) -> tensor<27x1x1xi64>
// CHECK-NEXT:  %16 = stablehlo.reshape %14 : (tensor<27xi64>) -> tensor<27x1xi64>
// CHECK-NEXT:  %17 = "stablehlo.gather"(%arg0, %16) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<53xf64>, tensor<27x1xi64>) -> tensor<27x1xf64>
// CHECK-NEXT:  %18 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<27xi1>) -> tensor<27x1x1xi1>
// CHECK-NEXT:  %19 = stablehlo.select %18, %15, %c : tensor<27x1x1xi1>, tensor<27x1x1xi64>
// CHECK-NEXT:  %20 = "stablehlo.scatter"(%12, %19, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:  stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<53xf64>, tensor<27x1x1xi64>, tensor<27x1xf64>) -> tensor<53xf64>
// CHECK-NEXT:  return %20 : tensor<53xf64>
// CHECK-NEXT:  }

// -----

// A read that only a guarded link uses is discarded with it: here the read
// steps into the next block exactly where the predicate fails, and would
// otherwise clash with that block's writer.
func.func @guarded_read(%x: tensor<40xf64>, %y: tensor<40xf64>) -> tensor<40xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c10 = stablehlo.constant dense<10> : tensor<i64>
  %0:2 = stablehlo.while(%b = %c0, %acc = %y) : tensor<i64>, tensor<40xf64>
   cond {
    %c = stablehlo.compare LT, %b, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %base = stablehlo.multiply %b, %c4 : tensor<i64>
    %i1 = stablehlo.reshape %base : (tensor<i64>) -> tensor<1x1xi64>
    %v1 = stablehlo.dynamic_slice %x, %b, sizes = [1] : (tensor<40xf64>, tensor<i64>) -> tensor<1xf64>
    %s1 = "stablehlo.scatter"(%acc, %i1, %v1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %bv: tensor<f64>):
      stablehlo.return %bv : tensor<f64>
    }) : (tensor<40xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<40xf64>
    %r = stablehlo.remainder %b, %c2 : tensor<i64>
    %p = stablehlo.compare EQ, %r, %c0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %own = stablehlo.add %base, %c2 : tensor<i64>
    %next = stablehlo.add %base, %c4 : tensor<i64>
    %at = stablehlo.select %p, %own, %next : tensor<i1>, tensor<i64>
    %v2 = stablehlo.dynamic_slice %s1, %at, sizes = [1] : (tensor<40xf64>, tensor<i64>) -> tensor<1xf64>
    %e2 = stablehlo.add %base, %c1 : tensor<i64>
    %i2 = stablehlo.reshape %e2 : (tensor<i64>) -> tensor<1x1xi64>
    %s2 = "stablehlo.scatter"(%s1, %i2, %v2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %bv: tensor<f64>):
      stablehlo.return %bv : tensor<f64>
    }) : (tensor<40xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<40xf64>
    %u = stablehlo.select %p, %s2, %s1 : tensor<i1>, tensor<40xf64>
    %bn = stablehlo.add %b, %c1 : tensor<i64>
    stablehlo.return %bn, %u : tensor<i64>, tensor<40xf64>
  }
  return %0#1 : tensor<40xf64>
}

// CHECK:  func.func @guarded_read(%arg0: tensor<40xf64>, %arg1: tensor<40xf64>) -> tensor<40xf64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<40> : tensor<10x1x1xi64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:  %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:  %c_3 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:  %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:  %1 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:  %2 = stablehlo.multiply %0, %1 : tensor<10xi64>
// CHECK-NEXT:  %3 = stablehlo.reshape %2 : (tensor<10xi64>) -> tensor<10x1x1xi64>
// CHECK-NEXT:  %4 = stablehlo.slice %arg0 [0:10] : (tensor<40xf64>) -> tensor<10xf64>
// CHECK-NEXT:  %5 = stablehlo.reshape %4 : (tensor<10xf64>) -> tensor<10x1xf64>
// CHECK-NEXT:  %6 = "stablehlo.scatter"(%arg1, %3, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:  stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<40xf64>, tensor<10x1x1xi64>, tensor<10x1xf64>) -> tensor<40xf64>
// CHECK-NEXT:  %7 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:  %8 = stablehlo.remainder %0, %7 : tensor<10xi64>
// CHECK-NEXT:  %9 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:  %10 = stablehlo.compare EQ, %8, %9 : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK-NEXT:  %11 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:  %12 = stablehlo.add %2, %11 : tensor<10xi64>
// CHECK-NEXT:  %13 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:  %14 = stablehlo.add %2, %13 : tensor<10xi64>
// CHECK-NEXT:  %15 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<10xi1>) -> tensor<10xi1>
// CHECK-NEXT:  %16 = stablehlo.select %15, %12, %14 : tensor<10xi1>, tensor<10xi64>
// CHECK-NEXT:  %17 = stablehlo.reshape %16 : (tensor<10xi64>) -> tensor<10x1xi64>
// CHECK-NEXT:  %18 = "stablehlo.gather"(%arg1, %17) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<40xf64>, tensor<10x1xi64>) -> tensor<10x1xf64>
// CHECK-NEXT:  %19 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:  %20 = stablehlo.add %2, %19 : tensor<10xi64>
// CHECK-NEXT:  %21 = stablehlo.reshape %20 : (tensor<10xi64>) -> tensor<10x1x1xi64>
// CHECK-NEXT:  %22 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<10xi1>) -> tensor<10x1x1xi1>
// CHECK-NEXT:  %23 = stablehlo.select %22, %21, %c : tensor<10x1x1xi1>, tensor<10x1x1xi64>
// CHECK-NEXT:  %24 = "stablehlo.scatter"(%6, %23, %18) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:  stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<40xf64>, tensor<10x1x1xi64>, tensor<10x1xf64>) -> tensor<40xf64>
// CHECK-NEXT:  return %24 : tensor<40xf64>
// CHECK-NEXT:  }

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
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %3 = stablehlo.multiply %0, %2 : tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<4xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<2xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:   %6 = stablehlo.add %4, %5 : tensor<4x2xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<4x2xi64>) -> tensor<4x2x1xi64>
// CHECK-NEXT:   %8 = "stablehlo.gather"(%arg0, %7) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x2x1xi64>) -> tensor<4x2x1xf64>
// CHECK-NEXT:   %9 = stablehlo.reshape %8 : (tensor<4x2x1xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:   %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x2xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %12 = stablehlo.add %11, %10 : tensor<4xf64>
// CHECK-NEXT:   %13 = stablehlo.reshape %12 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %14 = stablehlo.reshape %13 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %15 = stablehlo.dynamic_update_slice %arg1, %14, %c : (tensor<4xf64>, tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:   return %15 : tensor<4xf64>
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
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.dynamic_slice %arg0, %c_0, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:   %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %4 = stablehlo.add %cst, %3 : tensor<f64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %8 = stablehlo.dynamic_update_slice %arg0, %7, %c : (tensor<8xf64>, tensor<4xf64>, tensor<i64>) -> tensor<8xf64>
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
// finds the windows of two iterations one apart meet, and leaves the loop to
// the enumeration. That finds iteration i reads slot i+1 before any iteration
// writes it, so the read sees the entry buffer: batched.
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
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [1:33] : (tensor<33xf64>) -> tensor<32xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<32xf64>) -> tensor<32x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<32x1xf64>) -> tensor<32xf64>
// CHECK-NEXT:   %3 = stablehlo.dynamic_update_slice %arg0, %2, %c : (tensor<33xf64>, tensor<32xf64>, tensor<i64>) -> tensor<33xf64>
// CHECK-NEXT:   return %3 : tensor<33xf64>
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
