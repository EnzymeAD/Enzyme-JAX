// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A window of 4 by 4 of the [4, 1728] view, at column 432 i + 80: the window's
// rows and columns are two more dimensions of the grid, with the view's
// strides for scales, and the view factors as [4, 4, 432].
func.func @window_view(%arg: tensor<6912xf64>) -> tensor<4x4x4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c432 = stablehlo.constant dense<432> : tensor<4x1xi32>
  %c80 = stablehlo.constant dense<80> : tensor<4x1xi32>
  %i = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %m = stablehlo.multiply %i, %c432 : tensor<4x1xi32>
  %k = stablehlo.add %m, %c80 : tensor<4x1xi32>
  %idx = stablehlo.pad %k, %c0, low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<4x1xi32>, tensor<i32>) -> tensor<4x2xi32>
  %r = stablehlo.reshape %arg : (tensor<6912xf64>) -> tensor<1728x4xf64>
  %t = stablehlo.transpose %r, dims = [1, 0] : (tensor<1728x4xf64>) -> tensor<4x1728xf64>
  %g = "stablehlo.gather"(%t, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 4>}> : (tensor<4x1728xf64>, tensor<4x2xi32>) -> tensor<4x4x4xf64>
  return %g : tensor<4x4x4xf64>
}

// CHECK:    func.func @window_view(%arg0: tensor<6912xf64>) -> tensor<4x4x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<6912xf64>) -> tensor<1728x4xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1728x4xf64>) -> tensor<4x1728xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x1728xf64>) -> tensor<4x4x432xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:4, 0:4, 80:84] : (tensor<4x4x432xf64>) -> tensor<4x4x4xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0, 2] : (tensor<4x4x4xf64>) -> tensor<4x4x4xf64>
// CHECK-NEXT:    return %4 : tensor<4x4x4xf64>
// CHECK-NEXT:  }

// -----

// The window comes first in the result (offset dimension 0): the grid's
// result, windows last, is transposed into the gather's order.
func.func @window_leading(%x: tensor<6x8xf64>) -> tensor<6x4x2xf64> {
  %c2 = stablehlo.constant dense<2> : tensor<4x1xi32>
  %i = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %idx = stablehlo.multiply %i, %c2 : tensor<4x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 2>}> : (tensor<6x8xf64>, tensor<4x1xi32>) -> tensor<6x4x2xf64>
  return %g : tensor<6x4x2xf64>
}

// CHECK:    func.func @window_leading(%arg0: tensor<6x8xf64>) -> tensor<6x4x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<6x8xf64>) -> tensor<6x4x2xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 0, 2] : (tensor<6x4x2xf64>) -> tensor<4x6x2xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<4x6x2xf64>) -> tensor<6x4x2xf64>
// CHECK-NEXT:    return %2 : tensor<6x4x2xf64>
// CHECK-NEXT:  }

// -----

// A window along a one-dimensional operand: the rows 8 i of [8, 8].
func.func @window_1d(%x: tensor<64xf64>) -> tensor<4x8xf64> {
  %c8 = stablehlo.constant dense<8> : tensor<4x1xi32>
  %i = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %idx = stablehlo.multiply %i, %c8 : tensor<4x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8>}> : (tensor<64xf64>, tensor<4x1xi32>) -> tensor<4x8xf64>
  return %g : tensor<4x8xf64>
}

// CHECK:    func.func @window_1d(%arg0: tensor<64xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<64xf64>) -> tensor<8x8xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<8x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    return %2 : tensor<4x8xf64>
// CHECK-NEXT:  }

// -----

// A window at one index is a dynamic slice.
func.func @window_scalar(%x: tensor<64xf64>, %k: tensor<1xi32>) -> tensor<8xf64> {
  %g = "stablehlo.gather"(%x, %k) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = array<i64: 8>}> : (tensor<64xf64>, tensor<1xi32>) -> tensor<8xf64>
  return %g : tensor<8xf64>
}

// CHECK:    func.func @window_scalar(%arg0: tensor<64xf64>, %arg1: tensor<1xi32>) -> tensor<8xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<1xi32>) -> tensor<i32>
// CHECK-NEXT:    %1 = stablehlo.dynamic_slice %arg0, %0, sizes = [8] : (tensor<64xf64>, tensor<i32>) -> tensor<8xf64>
// CHECK-NEXT:    return %1 : tensor<8xf64>
// CHECK-NEXT:  }

// -----

// The last window would run past the end, where the gather clamps its
// start: left alone.
func.func @window_overrun(%x: tensor<64xf64>) -> tensor<9x8xf64> {
  %c8 = stablehlo.constant dense<8> : tensor<9x1xi32>
  %i = stablehlo.iota dim = 0 : tensor<9x1xi32>
  %idx = stablehlo.multiply %i, %c8 : tensor<9x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8>}> : (tensor<64xf64>, tensor<9x1xi32>) -> tensor<9x8xf64>
  return %g : tensor<9x8xf64>
}

// CHECK:    func.func @window_overrun(%arg0: tensor<64xf64>) -> tensor<9x8xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<9x1xi32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<9x1xi32>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %c : tensor<9x1xi32>
// CHECK-NEXT:    %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8>}> : (tensor<64xf64>, tensor<9x1xi32>) -> tensor<9x8xf64>
// CHECK-NEXT:    return %2 : tensor<9x8xf64>
// CHECK-NEXT:  }
