// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// y[idx[i]] = s for every i: one scatter of the broadcast scalar at all
// indices.
func.func @set_subvector(%idx: tensor<8xi64>, %y: tensor<32xf64>, %s: tensor<f64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %j, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @set_subvector(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>, %arg2: tensor<f64>) -> tensor<32xf64> {
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xi64>, tensor<8x1xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:   %4 = "stablehlo.scatter"(%arg1, %2, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %4 : tensor<32xf64>
// CHECK-NEXT: }

// -----

// y[2*i + 1] += x[i]: the index is computed from the induction variable and
// the update read from an invariant table.
func.func @strided_accumulate(%x: tensor<8xf64>, %y: tensor<32xf64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %vs = stablehlo.reshape %v : (tensor<1xf64>) -> tensor<f64>
    %m = stablehlo.multiply %i, %c2 : tensor<i64>
    %j = stablehlo.add %m, %c1 : tensor<i64>
    %jr = stablehlo.reshape %j : (tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %jr, %vs) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %r = stablehlo.add %a, %b : tensor<f64>
      stablehlo.return %r : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @strided_accumulate(%arg0: tensor<8xf64>, %arg1: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<8x1xi64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %5 = stablehlo.multiply %0, %4 : tensor<8xi64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %7 = stablehlo.add %5, %6 : tensor<8xi64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %9 = "stablehlo.scatter"(%arg1, %8, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     %10 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %10 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %9 : tensor<32xf64>
// CHECK-NEXT: }

// -----

// The carried buffer is read inside the loop: not batched.
func.func @reads_carried(%idx: tensor<8xi64>, %y: tensor<32xf64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %v = stablehlo.dynamic_slice %acc, %i, sizes = [1] : (tensor<32xf64>, tensor<i64>) -> tensor<1xf64>
    %vs = stablehlo.reshape %v : (tensor<1xf64>) -> tensor<f64>
    %u = "stablehlo.scatter"(%acc, %j, %vs) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @reads_carried(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xi64>, tensor<8x1xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %3 = stablehlo.reshape %0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %4 = "stablehlo.gather"(%arg1, %3) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<32xf64>, tensor<8x1xi64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %6 = "stablehlo.scatter"(%arg1, %2, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %6 : tensor<32xf64>
// CHECK-NEXT: }

// -----

// Without the parallel tag the iterations may not commute: not batched.
func.func @sequential(%idx: tensor<8xi64>, %y: tensor<32xf64>, %s: tensor<f64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %j, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @sequential(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>, %arg2: tensor<f64>) -> tensor<32xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg1) : tensor<i64>, tensor<32xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:     %2 = "stablehlo.scatter"(%iterArg_2, %1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %3, %2 : tensor<i64>, tensor<32xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<32xf64>
// CHECK-NEXT: }
