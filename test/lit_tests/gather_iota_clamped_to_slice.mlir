// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

// A gather of a 1-D operand at constant consecutive indices is the slice
// starting at the first index (`buf[lane + 4]` for every lane of an
// unrolled tree reduction step).
func.func @inbounds(%x: tensor<16xf64>) -> tensor<8xf64> {
  %idx = stablehlo.constant dense<[[4], [5], [6], [7], [8], [9], [10], [11]]> : tensor<8x1xi64>
  %r = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<8x1xi64>) -> tensor<8xf64>
  return %r : tensor<8xf64>
}

// CHECK:  func.func @inbounds(%arg0: tensor<16xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [4:12] : (tensor<16xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %0 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// Indices past the end clamp to the last element: the slice is followed by
// that many copies of it.
func.func @clamped(%x: tensor<8xf64>) -> tensor<8xf64> {
  %idx = stablehlo.constant dense<[[4], [5], [6], [7], [8], [9], [10], [11]]> : tensor<8x1xi64>
  %r = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<8x1xi64>) -> tensor<8xf64>
  return %r : tensor<8xf64>
}

// CHECK:  func.func @clamped(%arg0: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [7:8] : (tensor<8xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %arg0 [4:8] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %3 = stablehlo.concatenate %2, %1, dim = 0 : (tensor<4xf64>, tensor<4xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %3 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// Entirely past the end: every element is the last one.
func.func @all_clamped(%x: tensor<4xf64>) -> tensor<4xf64> {
  %idx = stablehlo.constant dense<[[4], [5], [6], [7]]> : tensor<4x1xi64>
  %r = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %r : tensor<4xf64>
}

// CHECK:  func.func @all_clamped(%arg0: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %1 : tensor<4xf64>
// CHECK-NEXT: }

// -----

// A strided run is a strided slice.
func.func @strided(%x: tensor<16xf64>) -> tensor<4xf64> {
  %idx = stablehlo.constant dense<[[0], [2], [4], [6]]> : tensor<4x1xi64>
  %r = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %r : tensor<4xf64>
}

// CHECK:  func.func @strided(%arg0: tensor<16xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:8:2] : (tensor<16xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %0 : tensor<4xf64>
// CHECK-NEXT: }
