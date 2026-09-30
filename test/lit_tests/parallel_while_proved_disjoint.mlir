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
