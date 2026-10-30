// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Iteration k of a loop over 6 writes lanes 36 i + 6 k of a buffer of 6 rows
// of 36: the grid 36 i factors the buffer as [6, 36] and the loop's counter,
// in [0, 6) by the loop's bounds, moves the update's start along the rows'
// 36: a dynamic update slice at [0, 6 k].
func.func @loop_scatter(%buf: tensor<216xf64>, %vals: tensor<6x6xf64>) -> tensor<216xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %c6v = stablehlo.constant dense<6> : tensor<6x1xi64>
  %c36 = stablehlo.constant dense<36> : tensor<6x1xi64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %buf) : tensor<i64>, tensor<216xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.broadcast_in_dim %k, dims = [] : (tensor<i64>) -> tensor<6x1xi64>
    %i = stablehlo.iota dim = 0 : tensor<6x1xi64>
    %0 = stablehlo.multiply %i, %c36 : tensor<6x1xi64>
    %1 = stablehlo.multiply %row, %c6v : tensor<6x1xi64>
    %idx = stablehlo.add %0, %1 : tensor<6x1xi64>
    %upd = stablehlo.dynamic_slice %vals, %k, %c0, sizes = [1, 6] : (tensor<6x6xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
    %upd1 = stablehlo.reshape %upd : (tensor<1x6xf64>) -> tensor<6xf64>
    %acc2 = "stablehlo.scatter"(%acc, %idx, %upd1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<216xf64>, tensor<6x1xi64>, tensor<6xf64>) -> tensor<216xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<216xf64>
  }
  return %r#1 : tensor<216xf64>
}

// CHECK:      func.func @loop_scatter(%arg0: tensor<216xf64>, %arg1: tensor<6x6xf64>) -> tensor<216xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<6> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<216xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg1, %iterArg, %c_1, sizes = [1, 6] : (tensor<6x6xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x6xf64>) -> tensor<6xf64>
// CHECK-NEXT:     %3 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %4 = stablehlo.multiply %3, %c : tensor<i32>
// CHECK-NEXT:     %5 = stablehlo.add %c_0, %4 : tensor<i32>
// CHECK-NEXT:     %6 = stablehlo.reshape %iterArg_4 : (tensor<216xf64>) -> tensor<6x36xf64>
// CHECK-NEXT:     %7 = stablehlo.reshape %2 : (tensor<6xf64>) -> tensor<6x1xf64>
// CHECK-NEXT:     %8 = stablehlo.transpose %7, dims = [0, 1] : (tensor<6x1xf64>) -> tensor<6x1xf64>
// CHECK-NEXT:     %9 = stablehlo.dynamic_update_slice %6, %8, %c_0, %5 : (tensor<6x36xf64>, tensor<6x1xf64>, tensor<i32>, tensor<i32>) -> tensor<6x36xf64>
// CHECK-NEXT:     %10 = stablehlo.reshape %9 : (tensor<6x36xf64>) -> tensor<216xf64>
// CHECK-NEXT:     %11 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %11, %10 : tensor<i64>, tensor<216xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<216xf64>
// CHECK-NEXT:  }

// -----

// Iteration k of a loop over 8 reads lanes 64 k + i, i < 64, of a vector of
// 512: a dynamic slice of the vector viewed as [8, 64] at row k.
func.func @loop_gather(%x: tensor<512xf64>) -> tensor<64xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %c64 = stablehlo.constant dense<64> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<64xf64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %zero) : tensor<i64>, tensor<64xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %off = stablehlo.multiply %k, %c64 : tensor<i64>
    %offb = stablehlo.broadcast_in_dim %off, dims = [] : (tensor<i64>) -> tensor<64x1xi64>
    %i = stablehlo.iota dim = 0 : tensor<64x1xi64>
    %idx = stablehlo.add %i, %offb : tensor<64x1xi64>
    %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<512xf64>, tensor<64x1xi64>) -> tensor<64xf64>
    %acc2 = stablehlo.add %acc, %g : tensor<64xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<64xf64>
  }
  return %r#1 : tensor<64xf64>
}

// CHECK:      func.func @loop_gather(%arg0: tensor<512xf64>) -> tensor<64xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<64> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<64xf64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<64xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %c : tensor<i32>
// CHECK-NEXT:     %3 = stablehlo.add %c_0, %2 : tensor<i32>
// CHECK-NEXT:     %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [64] : (tensor<512xf64>, tensor<i32>) -> tensor<64xf64>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<64xf64>) -> tensor<64x1xf64>
// CHECK-NEXT:     %6 = stablehlo.reshape %5 : (tensor<64x1xf64>) -> tensor<64xf64>
// CHECK-NEXT:     %7 = stablehlo.add %iterArg_4, %6 : tensor<64xf64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %8, %7 : tensor<i64>, tensor<64xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<64xf64>
// CHECK-NEXT:  }

// -----

// An offset with no provable range (a function argument): left alone.
func.func @unbounded(%x: tensor<512xf64>, %k: tensor<i64>) -> tensor<64xf64> {
  %c64 = stablehlo.constant dense<64> : tensor<i64>
  %off = stablehlo.multiply %k, %c64 : tensor<i64>
  %offb = stablehlo.broadcast_in_dim %off, dims = [] : (tensor<i64>) -> tensor<64x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<64x1xi64>
  %idx = stablehlo.add %i, %offb : tensor<64x1xi64>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<512xf64>, tensor<64x1xi64>) -> tensor<64xf64>
  return %g : tensor<64xf64>
}

// CHECK:      func.func @unbounded(%arg0: tensor<512xf64>, %arg1: tensor<i64>) -> tensor<64xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<64> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.multiply %arg1, %c : tensor<i64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<64x1xi64>
// CHECK-NEXT:   %2 = stablehlo.iota dim = 0 : tensor<64x1xi64>
// CHECK-NEXT:   %3 = stablehlo.add %2, %1 : tensor<64x1xi64>
// CHECK-NEXT:   %4 = "stablehlo.gather"(%arg0, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<512xf64>, tensor<64x1xi64>) -> tensor<64xf64>
// CHECK-NEXT:   return %4 : tensor<64xf64>
// CHECK-NEXT:  }

// -----

// The scalar is the loop's counter times 16: the constant is peeled off, and
// the counter, in [0, 6), moves the update's start along the rows of [6, 16].
func.func @scaled_scalar(%buf: tensor<96xf64>, %vals: tensor<6x4xf64>) -> tensor<96xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %buf) : tensor<i64>, tensor<96xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %s = stablehlo.multiply %k, %c16 : tensor<i64>
    %row = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<i64>) -> tensor<4x1xi64>
    %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %idx = stablehlo.add %i, %row : tensor<4x1xi64>
    %upd = stablehlo.dynamic_slice %vals, %k, %c0, sizes = [1, 4] : (tensor<6x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    %upd1 = stablehlo.reshape %upd : (tensor<1x4xf64>) -> tensor<4xf64>
    %acc2 = "stablehlo.scatter"(%acc, %idx, %upd1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<96xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<96xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<96xf64>
  }
  return %r#1 : tensor<96xf64>
}

// CHECK:      func.func @scaled_scalar(%arg0: tensor<96xf64>, %arg1: tensor<6x4xf64>) -> tensor<96xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<16> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<96xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg1, %iterArg, %c_1, sizes = [1, 4] : (tensor<6x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %4 = stablehlo.multiply %3, %c : tensor<i32>
// CHECK-NEXT:     %5 = stablehlo.add %c_0, %4 : tensor<i32>
// CHECK-NEXT:     %6 = stablehlo.reshape %2 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %9 = stablehlo.dynamic_update_slice %iterArg_4, %8, %5 : (tensor<96xf64>, tensor<4xf64>, tensor<i32>) -> tensor<96xf64>
// CHECK-NEXT:     %10 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %10, %9 : tensor<i64>, tensor<96xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<96xf64>
// CHECK-NEXT:  }

// -----

// The index reads row k of an iota (the form batch fission leaves an index
// computation it lifted out of the loop, read back per iteration): the row
// at zero plus k times the iota's scale along the rows, when k stays within
// the rows. Reading 16 k + i of a vector of 96 is a dynamic slice at [k, 0]
// of the vector as [6, 16].
func.func @lifted_index(%x: tensor<96xf64>) -> tensor<6x4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<4xi32>
  %init = stablehlo.constant dense<0.0> : tensor<6x4xf64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %init) : tensor<i64>, tensor<6x4xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.iota dim = 0 : tensor<6x4xi32>
    %ds = stablehlo.dynamic_slice %v, %k, %c0, sizes = [1, 4] : (tensor<6x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
    %row = stablehlo.reshape %ds : (tensor<1x4xi32>) -> tensor<4xi32>
    %0 = stablehlo.multiply %row, %c16 : tensor<4xi32>
    %i = stablehlo.iota dim = 0 : tensor<4xi32>
    %1 = stablehlo.add %0, %i : tensor<4xi32>
    %idx = stablehlo.reshape %1 : (tensor<4xi32>) -> tensor<4x1xi32>
    %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<96xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %g1 = stablehlo.reshape %g : (tensor<4xf64>) -> tensor<1x4xf64>
    %acc2 = stablehlo.dynamic_update_slice %acc, %g1, %k, %c0 : (tensor<6x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<6x4xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<6x4xf64>
  }
  return %r#1 : tensor<6x4xf64>
}

// CHECK:      func.func @lifted_index(%arg0: tensor<96xf64>) -> tensor<6x4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<16> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x4xf64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<6x4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %c : tensor<i32>
// CHECK-NEXT:     %3 = stablehlo.add %c_0, %2 : tensor<i32>
// CHECK-NEXT:     %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [4] : (tensor<96xf64>, tensor<i32>) -> tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.reshape %5 : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:     %7 = stablehlo.dynamic_update_slice %iterArg_4, %6, %iterArg, %c_1 : (tensor<6x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<6x4xf64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %8, %7 : tensor<i64>, tensor<6x4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<6x4xf64>
// CHECK-NEXT:  }

// -----

// The loop runs past the iota's rows, where the dynamic slice clamps its
// start: left alone.
func.func @lifted_index_clamped(%x: tensor<96xf64>) -> tensor<8x4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<4xi32>
  %init = stablehlo.constant dense<0.0> : tensor<8x4xf64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %init) : tensor<i64>, tensor<8x4xf64>
   cond {
    %c = stablehlo.compare LT, %k, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.iota dim = 0 : tensor<6x4xi32>
    %ds = stablehlo.dynamic_slice %v, %k, %c0, sizes = [1, 4] : (tensor<6x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
    %row = stablehlo.reshape %ds : (tensor<1x4xi32>) -> tensor<4xi32>
    %0 = stablehlo.multiply %row, %c16 : tensor<4xi32>
    %i = stablehlo.iota dim = 0 : tensor<4xi32>
    %1 = stablehlo.add %0, %i : tensor<4xi32>
    %idx = stablehlo.reshape %1 : (tensor<4xi32>) -> tensor<4x1xi32>
    %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<96xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %g1 = stablehlo.reshape %g : (tensor<4xf64>) -> tensor<1x4xf64>
    %acc2 = stablehlo.dynamic_update_slice %acc, %g1, %k, %c0 : (tensor<8x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<8x4xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<8x4xf64>
  }
  return %r#1 : tensor<8x4xf64>
}

// CHECK:      func.func @lifted_index_clamped(%arg0: tensor<96xf64>) -> tensor<8x4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<16> : tensor<4xi32>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %cst) : tensor<i64>, tensor<8x4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.iota dim = 0 : tensor<6x4xi32>
// CHECK-NEXT:     %2 = stablehlo.dynamic_slice %1, %iterArg, %c, sizes = [1, 4] : (tensor<6x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x4xi32>) -> tensor<4xi32>
// CHECK-NEXT:     %4 = stablehlo.multiply %3, %c_2 : tensor<4xi32>
// CHECK-NEXT:     %5 = stablehlo.iota dim = 0 : tensor<4xi32>
// CHECK-NEXT:     %6 = stablehlo.add %4, %5 : tensor<4xi32>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<4xi32>) -> tensor<4x1xi32>
// CHECK-NEXT:     %8 = "stablehlo.gather"(%arg0, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<96xf64>, tensor<4x1xi32>) -> tensor<4xf64>
// CHECK-NEXT:     %9 = stablehlo.reshape %8 : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:     %10 = stablehlo.dynamic_update_slice %iterArg_3, %9, %iterArg, %c : (tensor<8x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<8x4xf64>
// CHECK-NEXT:     %11 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %11, %10 : tensor<i64>, tensor<8x4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<8x4xf64>
// CHECK-NEXT:  }

// -----

// A one-element value computed from an iota is not a scalar the index
// carries: the index is read as it is, a grid from 3 by 2.
func.func @constant_scalar(%x: tensor<16xf64>) -> tensor<4xf64> {
  %c2 = stablehlo.constant dense<2> : tensor<4x1xi32>
  %io = stablehlo.iota dim = 0 : tensor<8xi32>
  %sl = stablehlo.slice %io [3:4] : (tensor<8xi32>) -> tensor<1xi32>
  %s = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32>
  %b = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<i32>) -> tensor<4x1xi32>
  %i = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %0 = stablehlo.multiply %i, %c2 : tensor<4x1xi32>
  %idx = stablehlo.add %0, %b : tensor<4x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi32>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK:      func.func @constant_scalar(%arg0: tensor<16xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [3:11:2] : (tensor<16xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %0 : tensor<4xf64>
// CHECK-NEXT:  }

// -----

// An index computed from floating-point values (the cell of each position)
// carries no scalar: the reading stops at the conversion.
func.func @float_index(%x: tensor<16xf64>, %pos: tensor<4xf32>) -> tensor<4xf64> {
  %c2 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %0 = stablehlo.multiply %pos, %c2 : tensor<4xf32>
  %1 = stablehlo.floor %0 : tensor<4xf32>
  %2 = stablehlo.convert %1 : (tensor<4xf32>) -> tensor<4xi32>
  %idx = stablehlo.reshape %2 : (tensor<4xi32>) -> tensor<4x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi32>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK:      func.func @float_index(%arg0: tensor<16xf64>, %arg1: tensor<4xf32>) -> tensor<4xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NEXT:   %0 = stablehlo.multiply %arg1, %cst : tensor<4xf32>
// CHECK-NEXT:   %1 = stablehlo.floor %0 : tensor<4xf32>
// CHECK-NEXT:   %2 = stablehlo.convert %1 : (tensor<4xf32>) -> tensor<4xi32>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<4xi32>) -> tensor<4x1xi32>
// CHECK-NEXT:   %4 = "stablehlo.gather"(%arg0, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi32>) -> tensor<4xf64>
// CHECK-NEXT:   return %4 : tensor<4xf64>
// CHECK-NEXT:  }

// -----

// In a loop tagged parallel, which the batching takes out, the scatter is
// left for the batched grid to fold as a whole.
func.func @parallel_loop(%buf: tensor<216xf64>, %vals: tensor<6x6xf64>) -> tensor<216xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %c6v = stablehlo.constant dense<6> : tensor<6x1xi64>
  %c36 = stablehlo.constant dense<36> : tensor<6x1xi64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %buf) : tensor<i64>, tensor<216xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.broadcast_in_dim %k, dims = [] : (tensor<i64>) -> tensor<6x1xi64>
    %i = stablehlo.iota dim = 0 : tensor<6x1xi64>
    %0 = stablehlo.multiply %i, %c36 : tensor<6x1xi64>
    %1 = stablehlo.multiply %row, %c6v : tensor<6x1xi64>
    %idx = stablehlo.add %0, %1 : tensor<6x1xi64>
    %upd = stablehlo.dynamic_slice %vals, %k, %c0, sizes = [1, 6] : (tensor<6x6xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
    %upd1 = stablehlo.reshape %upd : (tensor<1x6xf64>) -> tensor<6xf64>
    %acc2 = "stablehlo.scatter"(%acc, %idx, %upd1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<216xf64>, tensor<6x1xi64>, tensor<6xf64>) -> tensor<216xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<216xf64>
  }
  return %r#1 : tensor<216xf64>
}

// CHECK:      func.func @parallel_loop(%arg0: tensor<216xf64>, %arg1: tensor<6x6xf64>) -> tensor<216xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<6> : tensor<6x1xi64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<36> : tensor<6x1xi64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_4 = %arg0) : tensor<i64>, tensor<216xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<6x1xi64>
// CHECK-NEXT:     %2 = stablehlo.iota dim = 0 : tensor<6x1xi64>
// CHECK-NEXT:     %3 = stablehlo.multiply %2, %c_3 : tensor<6x1xi64>
// CHECK-NEXT:     %4 = stablehlo.multiply %1, %c_2 : tensor<6x1xi64>
// CHECK-NEXT:     %5 = stablehlo.add %3, %4 : tensor<6x1xi64>
// CHECK-NEXT:     %6 = stablehlo.dynamic_slice %arg1, %iterArg, %c, sizes = [1, 6] : (tensor<6x6xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<1x6xf64>) -> tensor<6xf64>
// CHECK-NEXT:     %8 = "stablehlo.scatter"(%iterArg_4, %5, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<216xf64>, tensor<6x1xi64>, tensor<6xf64>) -> tensor<216xf64>
// CHECK-NEXT:     %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %9, %8 : tensor<i64>, tensor<216xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<216xf64>
// CHECK-NEXT:  }
