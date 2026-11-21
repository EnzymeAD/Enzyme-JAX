// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The loop's index, broadcast and turned into scatter indices: the scatter
// writes the carried buffer, so nothing it feeds can leave the loop. Lifting
// the broadcast alone would only read it back from a hoisted iota per
// iteration: left as it is.
func.func @index_feeds_scatter(%buf: tensor<216xf64>, %vals: tensor<6x36xf64>) -> tensor<216xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %c36 = stablehlo.constant dense<36> : tensor<36x1xi64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %buf) : tensor<i64>, tensor<216xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.broadcast_in_dim %k, dims = [] : (tensor<i64>) -> tensor<36x1xi64>
    %lane = stablehlo.iota dim = 0 : tensor<36x1xi64>
    %0 = stablehlo.multiply %row, %c36 : tensor<36x1xi64>
    %idx = stablehlo.add %0, %lane : tensor<36x1xi64>
    %upd = stablehlo.dynamic_slice %vals, %k, %c0, sizes = [1, 36] : (tensor<6x36xf64>, tensor<i64>, tensor<i64>) -> tensor<1x36xf64>
    %upd1 = stablehlo.reshape %upd : (tensor<1x36xf64>) -> tensor<36xf64>
    %acc2 = "stablehlo.scatter"(%acc, %idx, %upd1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<216xf64>, tensor<36x1xi64>, tensor<36xf64>) -> tensor<216xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<216xf64>
  }
  return %r#1 : tensor<216xf64>
}

// CHECK:      func.func @index_feeds_scatter(%arg0: tensor<216xf64>, %arg1: tensor<6x36xf64>) -> tensor<216xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<36> : tensor<36x1xi64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0) : tensor<i64>, tensor<216xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<36x1xi64>
// CHECK-NEXT:     %2 = stablehlo.iota dim = 0 : tensor<36x1xi64>
// CHECK-NEXT:     %3 = stablehlo.multiply %1, %c_2 : tensor<36x1xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<36x1xi64>
// CHECK-NEXT:     %5 = stablehlo.dynamic_slice %arg1, %iterArg, %c, sizes = [1, 36] : (tensor<6x36xf64>, tensor<i64>, tensor<i64>) -> tensor<1x36xf64>
// CHECK-NEXT:     %6 = stablehlo.reshape %5 : (tensor<1x36xf64>) -> tensor<36xf64>
// CHECK-NEXT:     %7 = "stablehlo.scatter"(%iterArg_3, %4, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<216xf64>, tensor<36x1xi64>, tensor<36xf64>) -> tensor<216xf64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %8, %7 : tensor<i64>, tensor<216xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<216xf64>
// CHECK-NEXT: }

// -----


// The index broadcast into an elementwise computation on a row read from a
// value the loop does not write: the whole computation leaves the loop, so
// the broadcast is lifted with it.
func.func @index_feeds_elementwise(%vals: tensor<6x36xf64>) -> tensor<36xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<36xf64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %zero) : tensor<i64>, tensor<36xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %kf = stablehlo.convert %k : (tensor<i64>) -> tensor<f64>
    %w = stablehlo.broadcast_in_dim %kf, dims = [] : (tensor<f64>) -> tensor<36xf64>
    %row = stablehlo.dynamic_slice %vals, %k, %c0, sizes = [1, 36] : (tensor<6x36xf64>, tensor<i64>, tensor<i64>) -> tensor<1x36xf64>
    %row1 = stablehlo.reshape %row : (tensor<1x36xf64>) -> tensor<36xf64>
    %term = stablehlo.multiply %row1, %w : tensor<36xf64>
    %acc2 = stablehlo.add %acc, %term : tensor<36xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<36xf64>
  }
  return %r#1 : tensor<36xf64>
}

// CHECK:      func.func @index_feeds_elementwise(%arg0: tensor<6x36xf64>) -> tensor<36xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<6xf64>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<6xf64>
// CHECK-NEXT:   %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<36xf64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<6xf64>
// CHECK-NEXT:   %1 = stablehlo.multiply %cst_0, %cst_1 : tensor<6xf64>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<6xf64>
// CHECK-NEXT:   %3 = stablehlo.multiply %cst_0, %2 : tensor<6xf64>
// CHECK-NEXT:   %4 = stablehlo.add %3, %cst_1 : tensor<6xf64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<6xf64>) -> tensor<6x36xf64>
// CHECK-NEXT:   %6 = stablehlo.multiply %arg0, %5 : tensor<6x36xf64>
// CHECK-NEXT:   %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x36xf64>, tensor<f64>) -> tensor<36xf64>
// CHECK-NEXT:   %8 = stablehlo.add %7, %cst_2 : tensor<36xf64>
// CHECK-NEXT:   return %8 : tensor<36xf64>
// CHECK-NEXT: }
