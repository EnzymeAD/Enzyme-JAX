// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// ============================================================================
// Tests for gather with iota-like indexing that converts to slice
// ============================================================================

// Simple iota indexing: gather with iota indices should become a slice
func.func @gather_iota_to_slice(%arg0: tensor<10xi64>) -> tensor<5xi64> {
    %indices = stablehlo.iota dim = 0 : tensor<5x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<5x1xi64>) -> tensor<5xi64>
    return %0 : tensor<5xi64>
}
// CHECK-LABEL: func.func @gather_iota_to_slice
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [0:5]
// CHECK-NEXT: return %[[SLICE]]

// Iota with offset: gather with indices [2, 3, 4, 5] should become slice [2:6:1]
func.func @gather_iota_offset_to_slice(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %c = stablehlo.constant dense<2> : tensor<4x1xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %indices = stablehlo.add %iota, %c : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_iota_offset_to_slice
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [2:6]
// CHECK-NEXT: return %[[SLICE]]

// Iota with stride: gather with indices [0, 2, 4, 6] should become slice [0:8:2]
func.func @gather_iota_stride_to_slice(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %c = stablehlo.constant dense<2> : tensor<4x1xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %indices = stablehlo.multiply %iota, %c : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_iota_stride_to_slice
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [0:8:2]
// CHECK-NEXT: return %[[SLICE]]

// Iota with offset and stride: gather with indices [1, 3, 5, 7] should become slice [1:9:2]
func.func @gather_iota_offset_stride_to_slice(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %c_offset = stablehlo.constant dense<1> : tensor<4x1xi64>
    %c_scale = stablehlo.constant dense<2> : tensor<4x1xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %scaled = stablehlo.multiply %iota, %c_scale : tensor<4x1xi64>
    %indices = stablehlo.add %scaled, %c_offset : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_iota_offset_stride_to_slice
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [1:9:2]
// CHECK-NEXT: return %[[SLICE]]

// Constant iota-like indices: dense constant that forms an iota pattern
func.func @gather_const_iota_to_slice(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %indices = stablehlo.constant dense<[[0], [1], [2], [3]]> : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_const_iota_to_slice
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [0:4]
// CHECK-NEXT: return %[[SLICE]]

// ============================================================================
// Tests for gather with size-1 index -> dynamic_slice
// ============================================================================

// Size-1 index: scalar-like index should become dynamic_slice
func.func @gather_scalar_index_to_dynamic_slice(%arg0: tensor<10xi64>, %idx: tensor<1xi64>) -> tensor<1xi64> {
    %0 = "stablehlo.gather"(%arg0, %idx) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @gather_scalar_index_to_dynamic_slice
// CHECK: stablehlo.reshape
// CHECK: stablehlo.dynamic_slice

// Floating point elements in gather
func.func @gather_iota_float(%arg0: tensor<10xf64>) -> tensor<5xf64> {
    %indices = stablehlo.iota dim = 0 : tensor<5x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xf64>, tensor<5x1xi64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
}
// CHECK-LABEL: func.func @gather_iota_float
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [0:5]
// CHECK-NEXT: return %[[SLICE]]

// ============================================================================
// Negative tests: should NOT be simplified
// ============================================================================

// Non-1D operand: should not simplify (currently only supports 1D operands)
func.func @gather_non_1d_operand(%arg0: tensor<4x4xi64>) -> tensor<2xi64> {
    %indices = stablehlo.constant dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0, 1],
        start_index_map = [0, 1],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1, 1>
    } : (tensor<4x4xi64>, tensor<2x2xi64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
}
// CHECK-LABEL: func.func @gather_non_1d_operand
// CHECK: stablehlo.gather

// Slice sizes not all 1: should not simplify
func.func @gather_slice_size_not_1(%arg0: tensor<10xi64>) -> tensor<4x2xi64> {
    %indices = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 2>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4x2xi64>
    return %0 : tensor<4x2xi64>
}
// CHECK-LABEL: func.func @gather_slice_size_not_1
// CHECK: stablehlo.gather

// Non-iota indices: random indices should not simplify
func.func @gather_non_iota_indices(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %indices = stablehlo.constant dense<[[3], [1], [4], [2]]> : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_non_iota_indices
// CHECK: stablehlo.gather

// Negative stride iota: gather with indices [4, 3, 2, 1] should become slice + reverse
func.func @gather_iota_negative_stride_to_slice_reverse(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    // Indices: 4, 3, 2, 1 (start=4, stride=-1, count=4)
    // This is equivalent to: start + scale * iota = 4 + (-1) * [0, 1, 2, 3] = [4, 3, 2, 1]
    %c_offset = stablehlo.constant dense<4> : tensor<4x1xi64>
    %c_scale = stablehlo.constant dense<-1> : tensor<4x1xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %scaled = stablehlo.multiply %iota, %c_scale : tensor<4x1xi64>
    %indices = stablehlo.add %scaled, %c_offset : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_iota_negative_stride_to_slice_reverse
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [1:5]
// CHECK-NEXT: %[[REVERSE:.+]] = stablehlo.reverse %[[SLICE]], dims = [0]
// CHECK-NEXT: return %[[REVERSE]]

// Negative stride with offset: gather with indices [7, 5, 3, 1] (start=7, stride=-2)
func.func @gather_iota_negative_stride_offset_to_slice_reverse(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %c_offset = stablehlo.constant dense<7> : tensor<4x1xi64>
    %c_scale = stablehlo.constant dense<-2> : tensor<4x1xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4x1xi64>
    %scaled = stablehlo.multiply %iota, %c_scale : tensor<4x1xi64>
    %indices = stablehlo.add %scaled, %c_offset : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<10xi64>, tensor<4x1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}
// CHECK-LABEL: func.func @gather_iota_negative_stride_offset_to_slice_reverse
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [1:8:2]
// CHECK-NEXT: %[[REVERSE:.+]] = stablehlo.reverse %[[SLICE]], dims = [0]
// CHECK-NEXT: return %[[REVERSE]]
