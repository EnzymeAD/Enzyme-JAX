// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A CSR transpose padded to its longest row: the nested loop runs to a
// trip count read from the data, the same for every row, so it keeps
// running over the batched rows.
// M = max_i off[i+1] - off[i]; for i: acc = 0; for k < M: j = off[i] + k;
// if j < off[i+1]: acc += x[idx[j]]; y[i] = acc
func.func @csr_padded(%off: tensor<9xi32>, %idx: tensor<40xi32>, %x: tensor<40xf64>, %y: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %min = stablehlo.constant dense<-2147483648> : tensor<i32>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %lo = stablehlo.slice %off [0:8] : (tensor<9xi32>) -> tensor<8xi32>
  %hi = stablehlo.slice %off [1:9] : (tensor<9xi32>) -> tensor<8xi32>
  %len = stablehlo.subtract %hi, %lo : tensor<8xi32>
  %m = stablehlo.reduce(%len init: %min) applies stablehlo.maximum across dimensions = [0] : (tensor<8xi32>, tensor<i32>) -> tensor<i32>
  %m64 = stablehlo.convert %m : (tensor<i32>) -> tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %buf = %y) : tensor<i64>, tensor<8xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %l = stablehlo.dynamic_slice %off, %i, sizes = [1] : (tensor<9xi32>, tensor<i64>) -> tensor<1xi32>
    %ls = stablehlo.reshape %l : (tensor<1xi32>) -> tensor<i32>
    %i1 = stablehlo.add %i, %c1 : tensor<i64>
    %h = stablehlo.dynamic_slice %off, %i1, sizes = [1] : (tensor<9xi32>, tensor<i64>) -> tensor<1xi32>
    %hs = stablehlo.reshape %h : (tensor<1xi32>) -> tensor<i32>
    %1:2 = stablehlo.while(%k = %c0, %acc = %zero) : tensor<i64>, tensor<f64>
     cond {
      %c = stablehlo.compare LT, %k, %m64 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %k32 = stablehlo.convert %k : (tensor<i64>) -> tensor<i32>
      %j = stablehlo.add %ls, %k32 : tensor<i32>
      %in = stablehlo.compare LT, %j, %hs, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %j64 = stablehlo.convert %j : (tensor<i32>) -> tensor<i64>
      %jr = stablehlo.reshape %j64 : (tensor<i64>) -> tensor<1xi64>
      %col = "stablehlo.gather"(%idx, %jr) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<40xi32>, tensor<1xi64>) -> tensor<i32>
      %col64 = stablehlo.convert %col : (tensor<i32>) -> tensor<i64>
      %cr = stablehlo.reshape %col64 : (tensor<i64>) -> tensor<1xi64>
      %v = "stablehlo.gather"(%x, %cr) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<40xf64>, tensor<1xi64>) -> tensor<f64>
      %vm = stablehlo.select %in, %v, %zero : tensor<i1>, tensor<f64>
      %acc2 = stablehlo.add %acc, %vm : tensor<f64>
      %kn = stablehlo.add %k, %c1 : tensor<i64>
      stablehlo.return %kn, %acc2 : tensor<i64>, tensor<f64>
    }
    %ir = stablehlo.reshape %i : (tensor<i64>) -> tensor<1xi64>
    %b2 = "stablehlo.scatter"(%buf, %ir, %1#1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%p: tensor<f64>, %q: tensor<f64>):
      stablehlo.return %q : tensor<f64>
    }) : (tensor<8xf64>, tensor<1xi64>, tensor<f64>) -> tensor<8xf64>
    %in2 = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in2, %b2 : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @csr_padded(%arg0: tensor<9xi32>, %arg1: tensor<40xi32>, %arg2: tensor<40xf64>, %arg3: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<-2147483648> : tensor<i32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:8] : (tensor<9xi32>) -> tensor<8xi32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [1:9] : (tensor<9xi32>) -> tensor<8xi32>
// CHECK-NEXT:    %2 = stablehlo.subtract %1, %0 : tensor<8xi32>
// CHECK-NEXT:    %3 = stablehlo.reduce(%2 init: %c_1) applies stablehlo.maximum across dimensions = [0] : (tensor<8xi32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT:    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %5 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg0 [0:8] : (tensor<9xi32>) -> tensor<8xi32>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<8xi32>) -> tensor<8x1xi32>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<8x1xi32>) -> tensor<8xi32>
// CHECK-NEXT:    %9 = stablehlo.slice %arg0 [1:9] : (tensor<9xi32>) -> tensor<8xi32>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<8xi32>) -> tensor<8x1xi32>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<8x1xi32>) -> tensor<8xi32>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:    %13:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %12) : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %16 = stablehlo.compare LT, %iterArg, %4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %16 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:      %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i32>) -> tensor<8xi32>
// CHECK-NEXT:      %18 = stablehlo.add %8, %17 : tensor<8xi32>
// CHECK-NEXT:      %19 = stablehlo.compare LT, %18, %11, SIGNED : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
// CHECK-NEXT:      %20 = stablehlo.convert %18 : (tensor<8xi32>) -> tensor<8xi64>
// CHECK-NEXT:      %21 = stablehlo.reshape %20 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:      %22 = "stablehlo.gather"(%arg1, %21) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<40xi32>, tensor<8x1xi64>) -> tensor<8xi32>
// CHECK-NEXT:      %23 = stablehlo.convert %22 : (tensor<8xi32>) -> tensor<8xi64>
// CHECK-NEXT:      %24 = stablehlo.reshape %23 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:      %25 = "stablehlo.gather"(%arg2, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<40xf64>, tensor<8x1xi64>) -> tensor<8xf64>
// CHECK-NEXT:      %26 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:      %27 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<8xi1>) -> tensor<8xi1>
// CHECK-NEXT:      %28 = stablehlo.select %27, %25, %26 : tensor<8xi1>, tensor<8xf64>
// CHECK-NEXT:      %29 = stablehlo.add %iterArg_2, %28 : tensor<8xf64>
// CHECK-NEXT:      %30 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %30, %29 : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %14 = stablehlo.reshape %5 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:    %15 = "stablehlo.scatter"(%arg3, %14, %13#1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<8xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<8xf64>
// CHECK-NEXT:    return %15 : tensor<8xf64>
// CHECK-NEXT:  }
