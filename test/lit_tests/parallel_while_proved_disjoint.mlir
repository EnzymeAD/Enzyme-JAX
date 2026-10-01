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
// it cancels between iterations, and the rows stay disjoint.
func.func @scalar_base_plus_constant(%acc: tensor<12xf64>, %contrib: tensor<3xf64>, %s: tensor<i64>) -> tensor<12xf64> {
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

// CHECK:  func.func @scalar_base_plus_constant(%arg0: tensor<12xf64>, %arg1: tensor<3xf64>, %arg2: tensor<i64>) -> tensor<12xf64> {
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
// CHECK-NEXT:   %8 = "stablehlo.gather"(%arg0, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<12xf64>, tensor<2x3x1xi64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:   %10 = stablehlo.add %8, %9 : tensor<2x3xf64>
// CHECK-NEXT:   %11 = "stablehlo.scatter"(%arg0, %7, %10) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<12xf64>, tensor<2x3x1xi64>, tensor<2x3xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %11 : tensor<12xf64>
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
