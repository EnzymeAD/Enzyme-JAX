// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_gather" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A slice along the gather's batch dimensions is a slice of the indices.
func.func @batch(%x: tensor<100xf64>, %idx: tensor<20x32x1xi64>) -> tensor<20x4xf64> {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %s = stablehlo.slice %g [0:20, 0:4] : (tensor<20x32xf64>) -> tensor<20x4xf64>
  return %s : tensor<20x4xf64>
}

// CHECK:  func.func @batch(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> tensor<20x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:4, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x4x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x4x1xi64>) -> tensor<20x4xf64>
// CHECK-NEXT:    return %1 : tensor<20x4xf64>
// CHECK-NEXT:  }

// -----

// A batching pair: the operand is sliced along its paired dimension too.
func.func @batching(%x: tensor<20x100xf64>, %idx: tensor<20x32x1xi64>) -> tensor<8x32xf64> {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %s = stablehlo.slice %g [4:12, 0:32] : (tensor<20x32xf64>) -> tensor<8x32xf64>
  return %s : tensor<8x32xf64>
}

// CHECK:  func.func @batching(%arg0: tensor<20x100xf64>, %arg1: tensor<20x32x1xi64>) -> tensor<8x32xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [4:12, 0:32, 0:1] : (tensor<20x32x1xi64>) -> tensor<8x32x1xi64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [4:12, 0:100] : (tensor<20x100xf64>) -> tensor<8x100xf64>
// CHECK-NEXT:    %2 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x100xf64>, tensor<8x32x1xi64>) -> tensor<8x32xf64>
// CHECK-NEXT:    return %2 : tensor<8x32xf64>
// CHECK-NEXT:  }

// -----

// Two slices of one gather: each gets its own, the whole never computed.
func.func @two(%x: tensor<100xf64>, %idx: tensor<20x32x1xi64>) -> (tensor<20x4xf64>, tensor<20x2xf64>) {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %s = stablehlo.slice %g [0:20, 0:4] : (tensor<20x32xf64>) -> tensor<20x4xf64>
  %t = stablehlo.slice %g [0:20, 8:10] : (tensor<20x32xf64>) -> tensor<20x2xf64>
  return %s, %t : tensor<20x4xf64>, tensor<20x2xf64>
}

// CHECK:  func.func @two(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> (tensor<20x4xf64>, tensor<20x2xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:4, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x4x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x4x1xi64>) -> tensor<20x4xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg1 [0:20, 8:10, 0:1] : (tensor<20x32x1xi64>) -> tensor<20x2x1xi64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg0, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x2x1xi64>) -> tensor<20x2xf64>
// CHECK-NEXT:    return %1, %3 : tensor<20x4xf64>, tensor<20x2xf64>
// CHECK-NEXT:  }

// -----

// A slice along an offset dimension reads part of each window: left alone.
func.func @offset(%x: tensor<100x8xf64>, %idx: tensor<20x1xi64>) -> tensor<20x4xf64> {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<100x8xf64>, tensor<20x1xi64>) -> tensor<20x8xf64>
  %s = stablehlo.slice %g [0:20, 0:4] : (tensor<20x8xf64>) -> tensor<20x4xf64>
  return %s : tensor<20x4xf64>
}

// CHECK:  func.func @offset(%arg0: tensor<100x8xf64>, %arg1: tensor<20x1xi64>) -> tensor<20x4xf64> {
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<100x8xf64>, tensor<20x1xi64>) -> tensor<20x8xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:20, 0:4] : (tensor<20x8xf64>) -> tensor<20x4xf64>
// CHECK-NEXT:    return %1 : tensor<20x4xf64>
// CHECK-NEXT:  }

// -----

// The two slices make up the whole: gathering them apart gathers as much.
func.func @whole(%x: tensor<100xf64>, %idx: tensor<20x32x1xi64>) -> (tensor<20x16xf64>, tensor<20x16xf64>) {
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
  %a = stablehlo.slice %g [0:20, 0:16] : (tensor<20x32xf64>) -> tensor<20x16xf64>
  %b = stablehlo.slice %g [0:20, 16:32] : (tensor<20x32xf64>) -> tensor<20x16xf64>
  return %a, %b : tensor<20x16xf64>, tensor<20x16xf64>
}

// CHECK:  func.func @whole(%arg0: tensor<100xf64>, %arg1: tensor<20x32x1xi64>) -> (tensor<20x16xf64>, tensor<20x16xf64>) {
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<20x32x1xi64>) -> tensor<20x32xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:20, 0:16] : (tensor<20x32xf64>) -> tensor<20x16xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [0:20, 16:32] : (tensor<20x32xf64>) -> tensor<20x16xf64>
// CHECK-NEXT:    return %1, %2 : tensor<20x16xf64>, tensor<20x16xf64>
// CHECK-NEXT:  }
