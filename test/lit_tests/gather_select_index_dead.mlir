// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_select_index_dead" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_select_index_dead;gather_op_canon<16>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=FOLD

// The raised form of a guarded read in a grid-stride loop: lane l of block
// i reads x[(8 i + u) 256 + l] with u the number of guards that hold, and
// the value is only kept where both hold. Where they do, u is 3: the index
// is 256 (8 i + 3) + l, a grid, and the gather a slice of x as [3, 8, 256].
func.func @guarded_read(%x: tensor<6144xf64>) -> tensor<3x256xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<3x256xf64>
  %n = stablehlo.constant dense<4200> : tensor<3x256xi32>
  %c1 = stablehlo.constant dense<1> : tensor<3x256xi32>
  %c2 = stablehlo.constant dense<2> : tensor<3x256xi32>
  %c256 = stablehlo.constant dense<256> : tensor<3x256xi32>
  %c2048 = stablehlo.constant dense<2048> : tensor<3xi32>
  %c8 = stablehlo.constant dense<8> : tensor<3xi32>
  %i = stablehlo.iota dim = 0 : tensor<3xi32>
  %l = stablehlo.iota dim = 1 : tensor<3x256xi32>
  %0 = stablehlo.multiply %i, %c2048 : tensor<3xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
  %2 = stablehlo.add %1, %l : tensor<3x256xi32>
  %p1 = stablehlo.compare LT, %2, %n : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
  %3 = stablehlo.add %2, %c256 : tensor<3x256xi32>
  %q = stablehlo.compare LT, %3, %n : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
  %p2 = stablehlo.and %p1, %q : tensor<3x256xi1>
  %u1 = stablehlo.select %p1, %c2, %c1 : tensor<3x256xi1>, tensor<3x256xi32>
  %u1p = stablehlo.add %u1, %c1 : tensor<3x256xi32>
  %u = stablehlo.select %p2, %u1p, %u1 : tensor<3x256xi1>, tensor<3x256xi32>
  %4 = stablehlo.multiply %i, %c8 : tensor<3xi32>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
  %6 = stablehlo.add %u, %5 : tensor<3x256xi32>
  %7 = stablehlo.multiply %6, %c256 : tensor<3x256xi32>
  %8 = stablehlo.add %7, %l : tensor<3x256xi32>
  %idx = stablehlo.reshape %8 : (tensor<3x256xi32>) -> tensor<3x256x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6144xf64>, tensor<3x256x1xi32>) -> tensor<3x256xf64>
  %sq = stablehlo.multiply %g, %g : tensor<3x256xf64>
  %r = stablehlo.select %p2, %sq, %zero : tensor<3x256xi1>, tensor<3x256xf64>
  return %r : tensor<3x256xf64>
}

// CHECK:    func.func @guarded_read(%arg0: tensor<6144xf64>) -> tensor<3x256xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x256xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<4200> : tensor<3x256xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<3x256xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<2> : tensor<3x256xi32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<256> : tensor<3x256xi32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<2048> : tensor<3xi32>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<8> : tensor<3xi32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi32>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 1 : tensor<3x256xi32>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %c_3 : tensor<3xi32>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// CHECK-NEXT:    %4 = stablehlo.add %3, %1 : tensor<3x256xi32>
// CHECK-NEXT:    %5 = stablehlo.compare LT, %4, %c : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
// CHECK-NEXT:    %6 = stablehlo.add %4, %c_2 : tensor<3x256xi32>
// CHECK-NEXT:    %7 = stablehlo.compare LT, %6, %c : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
// CHECK-NEXT:    %8 = stablehlo.and %5, %7 : tensor<3x256xi1>
// CHECK-NEXT:    %9 = stablehlo.multiply %0, %c_4 : tensor<3xi32>
// CHECK-NEXT:    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// CHECK-NEXT:    %11 = stablehlo.add %c_1, %c_0 : tensor<3x256xi32>
// CHECK-NEXT:    %12 = stablehlo.add %11, %10 : tensor<3x256xi32>
// CHECK-NEXT:    %13 = stablehlo.multiply %12, %c_2 : tensor<3x256xi32>
// CHECK-NEXT:    %14 = stablehlo.add %13, %1 : tensor<3x256xi32>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<3x256xi32>) -> tensor<3x256x1xi32>
// CHECK-NEXT:    %16 = "stablehlo.gather"(%arg0, %15) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6144xf64>, tensor<3x256x1xi32>) -> tensor<3x256xf64>
// CHECK-NEXT:    %17 = stablehlo.multiply %16, %16 : tensor<3x256xf64>
// CHECK-NEXT:    %18 = stablehlo.select %8, %17, %cst : tensor<3x256xi1>, tensor<3x256xf64>
// CHECK-NEXT:    return %18 : tensor<3x256xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @guarded_read(%arg0: tensor<6144xf64>) -> tensor<3x256xf64> {
// FOLD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x256xf64>
// FOLD-NEXT:    %c = stablehlo.constant dense<4200> : tensor<3x256xi32>
// FOLD-NEXT:    %c_0 = stablehlo.constant dense<256> : tensor<3x256xi32>
// FOLD-NEXT:    %c_1 = stablehlo.constant dense<2048> : tensor<3xi32>
// FOLD-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi32>
// FOLD-NEXT:    %1 = stablehlo.iota dim = 1 : tensor<3x256xi32>
// FOLD-NEXT:    %2 = stablehlo.multiply %0, %c_1 : tensor<3xi32>
// FOLD-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// FOLD-NEXT:    %4 = stablehlo.add %3, %1 : tensor<3x256xi32>
// FOLD-NEXT:    %5 = stablehlo.compare LT, %4, %c : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
// FOLD-NEXT:    %6 = stablehlo.add %4, %c_0 : tensor<3x256xi32>
// FOLD-NEXT:    %7 = stablehlo.compare LT, %6, %c : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
// FOLD-NEXT:    %8 = stablehlo.and %5, %7 : tensor<3x256xi1>
// FOLD-NEXT:    %9 = stablehlo.reshape %arg0 : (tensor<6144xf64>) -> tensor<3x2048xf64>
// FOLD-NEXT:    %10 = stablehlo.slice %9 [0:3, 768:1024] : (tensor<3x2048xf64>) -> tensor<3x256xf64>
// FOLD-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<3x256xf64>) -> tensor<3x256xf64>
// FOLD-NEXT:    %12 = stablehlo.multiply %11, %11 : tensor<3x256xf64>
// FOLD-NEXT:    %13 = stablehlo.select %8, %12, %cst : tensor<3x256xi1>, tensor<3x256xf64>
// FOLD-NEXT:    return %13 : tensor<3x256xf64>
// FOLD-NEXT:  }

// -----

// The value is also read where the guard fails: the index stays.
func.func @read_outside(%x: tensor<6144xf64>) -> (tensor<3x256xf64>, tensor<f64>) {
  %zero = stablehlo.constant dense<0.0> : tensor<3x256xf64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %n = stablehlo.constant dense<4200> : tensor<3x256xi32>
  %c1 = stablehlo.constant dense<1> : tensor<3x256xi32>
  %c2 = stablehlo.constant dense<2> : tensor<3x256xi32>
  %c256 = stablehlo.constant dense<256> : tensor<3x256xi32>
  %c2048 = stablehlo.constant dense<2048> : tensor<3xi32>
  %c8 = stablehlo.constant dense<8> : tensor<3xi32>
  %i = stablehlo.iota dim = 0 : tensor<3xi32>
  %l = stablehlo.iota dim = 1 : tensor<3x256xi32>
  %0 = stablehlo.multiply %i, %c2048 : tensor<3xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
  %2 = stablehlo.add %1, %l : tensor<3x256xi32>
  %p1 = stablehlo.compare LT, %2, %n : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
  %u = stablehlo.select %p1, %c2, %c1 : tensor<3x256xi1>, tensor<3x256xi32>
  %4 = stablehlo.multiply %i, %c8 : tensor<3xi32>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
  %6 = stablehlo.add %u, %5 : tensor<3x256xi32>
  %7 = stablehlo.multiply %6, %c256 : tensor<3x256xi32>
  %8 = stablehlo.add %7, %l : tensor<3x256xi32>
  %idx = stablehlo.reshape %8 : (tensor<3x256xi32>) -> tensor<3x256x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6144xf64>, tensor<3x256x1xi32>) -> tensor<3x256xf64>
  %r = stablehlo.select %p1, %g, %zero : tensor<3x256xi1>, tensor<3x256xf64>
  %s = stablehlo.reduce(%g init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x256xf64>, tensor<f64>) -> tensor<f64>
  return %r, %s : tensor<3x256xf64>, tensor<f64>
}

// CHECK:    func.func @read_outside(%arg0: tensor<6144xf64>) -> (tensor<3x256xf64>, tensor<f64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x256xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<4200> : tensor<3x256xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<3x256xi32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<2> : tensor<3x256xi32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<256> : tensor<3x256xi32>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<2048> : tensor<3xi32>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<8> : tensor<3xi32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi32>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 1 : tensor<3x256xi32>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %c_4 : tensor<3xi32>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// CHECK-NEXT:    %4 = stablehlo.add %3, %1 : tensor<3x256xi32>
// CHECK-NEXT:    %5 = stablehlo.compare LT, %4, %c : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
// CHECK-NEXT:    %6 = stablehlo.select %5, %c_2, %c_1 : tensor<3x256xi1>, tensor<3x256xi32>
// CHECK-NEXT:    %7 = stablehlo.multiply %0, %c_5 : tensor<3xi32>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// CHECK-NEXT:    %9 = stablehlo.add %6, %8 : tensor<3x256xi32>
// CHECK-NEXT:    %10 = stablehlo.multiply %9, %c_3 : tensor<3x256xi32>
// CHECK-NEXT:    %11 = stablehlo.add %10, %1 : tensor<3x256xi32>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<3x256xi32>) -> tensor<3x256x1xi32>
// CHECK-NEXT:    %13 = "stablehlo.gather"(%arg0, %12) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6144xf64>, tensor<3x256x1xi32>) -> tensor<3x256xf64>
// CHECK-NEXT:    %14 = stablehlo.select %5, %13, %cst : tensor<3x256xi1>, tensor<3x256xf64>
// CHECK-NEXT:    %15 = stablehlo.reduce(%13 init: %cst_0) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x256xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    return %14, %15 : tensor<3x256xf64>, tensor<f64>
// CHECK-NEXT:  }

// FOLD:    func.func @read_outside(%arg0: tensor<6144xf64>) -> (tensor<3x256xf64>, tensor<f64>) {
// FOLD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x256xf64>
// FOLD-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FOLD-NEXT:    %c = stablehlo.constant dense<4200> : tensor<3x256xi32>
// FOLD-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<3x256xi32>
// FOLD-NEXT:    %c_2 = stablehlo.constant dense<2> : tensor<3x256xi32>
// FOLD-NEXT:    %c_3 = stablehlo.constant dense<256> : tensor<3x256xi32>
// FOLD-NEXT:    %c_4 = stablehlo.constant dense<2048> : tensor<3xi32>
// FOLD-NEXT:    %c_5 = stablehlo.constant dense<8> : tensor<3xi32>
// FOLD-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi32>
// FOLD-NEXT:    %1 = stablehlo.iota dim = 1 : tensor<3x256xi32>
// FOLD-NEXT:    %2 = stablehlo.multiply %0, %c_4 : tensor<3xi32>
// FOLD-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// FOLD-NEXT:    %4 = stablehlo.add %3, %1 : tensor<3x256xi32>
// FOLD-NEXT:    %5 = stablehlo.compare LT, %4, %c : (tensor<3x256xi32>, tensor<3x256xi32>) -> tensor<3x256xi1>
// FOLD-NEXT:    %6 = stablehlo.select %5, %c_2, %c_1 : tensor<3x256xi1>, tensor<3x256xi32>
// FOLD-NEXT:    %7 = stablehlo.multiply %0, %c_5 : tensor<3xi32>
// FOLD-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<3xi32>) -> tensor<3x256xi32>
// FOLD-NEXT:    %9 = stablehlo.add %6, %8 : tensor<3x256xi32>
// FOLD-NEXT:    %10 = stablehlo.multiply %9, %c_3 : tensor<3x256xi32>
// FOLD-NEXT:    %11 = stablehlo.add %10, %1 : tensor<3x256xi32>
// FOLD-NEXT:    %12 = stablehlo.reshape %11 : (tensor<3x256xi32>) -> tensor<3x256x1xi32>
// FOLD-NEXT:    %13 = "stablehlo.gather"(%arg0, %12) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6144xf64>, tensor<3x256x1xi32>) -> tensor<3x256xf64>
// FOLD-NEXT:    %14 = stablehlo.select %5, %13, %cst : tensor<3x256xi1>, tensor<3x256xf64>
// FOLD-NEXT:    %15 = stablehlo.reduce(%13 init: %cst_0) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x256xf64>, tensor<f64>) -> tensor<f64>
// FOLD-NEXT:    return %14, %15 : tensor<3x256xf64>, tensor<f64>
// FOLD-NEXT:  }
