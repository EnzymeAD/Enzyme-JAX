// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Each iteration fills a scratch row (positions 0..2) with values, reads them
// back as a window, and writes the sum into its own slot of the output. The
// scratch is dead after the loop: each iteration may have its own copy.
func.func @private_scratch(%out: tensor<4xf64>, %vals: tensor<4x3xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %scratch0 = stablehlo.constant dense<0.0> : tensor<8xf64>
  %widx = stablehlo.constant dense<[[0], [1], [2]]> : tensor<3x1xi64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %out, %scr = %scratch0) : tensor<i64>, tensor<4xf64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %vals, %i, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
    %r = stablehlo.reshape %row : (tensor<1x3xf64>) -> tensor<3xf64>
    %w = "stablehlo.scatter"(%scr, %widx, %r) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
    %win = stablehlo.slice %w [0:3] : (tensor<8xf64>) -> tensor<3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sum = stablehlo.reduce(%win init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %s1 = stablehlo.reshape %sum : (tensor<f64>) -> tensor<1xf64>
    %oidx = stablehlo.reshape %i : (tensor<i64>) -> tensor<1x1xi64>
    %nacc = "stablehlo.scatter"(%acc, %oidx, %s1) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %nacc, %w : tensor<i64>, tensor<4xf64>, tensor<8xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:    func.func @private_scratch(%arg0: tensor<4xf64>, %arg1: tensor<4x3xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0], [1], [2]]> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<4x3xf64>) -> tensor<4x1x3xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x1x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %c, dims = [1, 2] : (tensor<3x1xi64>) -> tensor<4x3x1xi64>
// CHECK-NEXT:    %4 = "stablehlo.scatter"(%cst, %3, %2) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4x8xf64>, tensor<4x3x1xi64>, tensor<4x3xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %4 [0:4, 0:3] : (tensor<4x8xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %8 = stablehlo.reshape %0 : (tensor<4xi64>) -> tensor<4x1x1xi64>
// CHECK-NEXT:    %9 = "stablehlo.scatter"(%arg0, %8, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4xf64>, tensor<4x1x1xi64>, tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %9 : tensor<4xf64>
// CHECK-NEXT:  }

// -----
// The same loop, but the window it reads back is wider than what the
// iteration wrote: positions 3 and 4 are written by no iteration, so they
// hold the loop's operand in every iteration, and each iteration's copy is
// seeded with it.
func.func @reads_more_than_it_wrote(%out: tensor<4xf64>, %vals: tensor<4x3xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %scratch0 = stablehlo.constant dense<0.0> : tensor<8xf64>
  %widx = stablehlo.constant dense<[[0], [1], [2]]> : tensor<3x1xi64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %out, %scr = %scratch0) : tensor<i64>, tensor<4xf64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %vals, %i, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
    %r = stablehlo.reshape %row : (tensor<1x3xf64>) -> tensor<3xf64>
    %w = "stablehlo.scatter"(%scr, %widx, %r) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
    %win5 = stablehlo.slice %w [0:5] : (tensor<8xf64>) -> tensor<5xf64>
    %win = stablehlo.slice %win5 [0:3] : (tensor<5xf64>) -> tensor<3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sum = stablehlo.reduce(%win init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %s1 = stablehlo.reshape %sum : (tensor<f64>) -> tensor<1xf64>
    %oidx = stablehlo.reshape %i : (tensor<i64>) -> tensor<1x1xi64>
    %nacc = "stablehlo.scatter"(%acc, %oidx, %s1) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %nacc, %w : tensor<i64>, tensor<4xf64>, tensor<8xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @reads_more_than_it_wrote(%arg0: tensor<4xf64>, %arg1: tensor<4x3xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<8xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0], [1], [2]]> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %arg1 : (tensor<4x3xf64>) -> tensor<4x1x3xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<4x1x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %c, dims = [1, 2] : (tensor<3x1xi64>) -> tensor<4x3x1xi64>
// CHECK-NEXT:    %5 = "stablehlo.scatter"(%1, %4, %3) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4x8xf64>, tensor<4x3x1xi64>, tensor<4x3xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %5 [0:4, 0:5] : (tensor<4x8xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:    %7 = stablehlo.slice %6 [0:4, 0:3] : (tensor<4x5xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %10 = stablehlo.reshape %0 : (tensor<4xi64>) -> tensor<4x1x1xi64>
// CHECK-NEXT:    %11 = "stablehlo.scatter"(%arg0, %10, %9) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4xf64>, tensor<4x1x1xi64>, tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %11 : tensor<4xf64>
// CHECK-NEXT:  }

// -----

// The same, but every iteration also writes position 3 after reading the
// window: a later iteration reads what the one before left there, so the
// scratch is not private and the loop is kept.
func.func @reads_what_the_last_iteration_wrote(%out: tensor<4xf64>, %vals: tensor<4x3xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %scratch0 = stablehlo.constant dense<0.0> : tensor<8xf64>
  %widx = stablehlo.constant dense<[[0], [1], [2]]> : tensor<3x1xi64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %out, %scr = %scratch0) : tensor<i64>, tensor<4xf64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %vals, %i, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
    %r = stablehlo.reshape %row : (tensor<1x3xf64>) -> tensor<3xf64>
    %w = "stablehlo.scatter"(%scr, %widx, %r) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
    %win5 = stablehlo.slice %w [0:5] : (tensor<8xf64>) -> tensor<5xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sum = stablehlo.reduce(%win5 init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<5xf64>, tensor<f64>) -> tensor<f64>
    %s1 = stablehlo.reshape %sum : (tensor<f64>) -> tensor<1xf64>
    %w2 = stablehlo.dynamic_update_slice %w, %s1, %c3 : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %oidx = stablehlo.reshape %i : (tensor<i64>) -> tensor<1x1xi64>
    %nacc = "stablehlo.scatter"(%acc, %oidx, %s1) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %nacc, %w2 : tensor<i64>, tensor<4xf64>, tensor<8xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @reads_what_the_last_iteration_wrote(%arg0: tensor<4xf64>, %arg1: tensor<4x3xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<8xf64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<{{\[\[}}0], [1], [2]]> : tensor<3x1xi64>
// CHECK-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_5 = %arg0, %iterArg_6 = %cst_3) : tensor<i64>, tensor<4xf64>, tensor<8xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %1 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.dynamic_slice %arg1, %iterArg, %c, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %2 = stablehlo.reshape %1 : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:      %3 = "stablehlo.scatter"(%iterArg_6, %c_4, %2) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT:      ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:      }) : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
// CHECK-NEXT:      %4 = stablehlo.slice %3 [0:5] : (tensor<8xf64>) -> tensor<5xf64>
// CHECK-NEXT:      %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<5xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %6 = stablehlo.reshape %5 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %3, %6, %c_1 : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:      %8 = stablehlo.reshape %iterArg : (tensor<i64>) -> tensor<1x1xi64>
// CHECK-NEXT:      %9 = "stablehlo.scatter"(%iterArg_5, %8, %6) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT:      ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:      }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
// CHECK-NEXT:      %10 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %10, %9, %7 : tensor<i64>, tensor<4xf64>, tensor<8xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1 : tensor<4xf64>
// CHECK-NEXT:  }
