// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: stablehlo-translate --interpret --allow-unregistered-dialect %s

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

// Gather with reshaped multi-dimensional iota indices
func.func @gather_reshaped_iota_to_slice(%arg0: tensor<10xi64>) -> tensor<4xi64> {
    %iota1 = stablehlo.iota dim = 0 : tensor<2x2xi64>
    %iota2 = stablehlo.iota dim = 1 : tensor<2x2xi64>
    %c2 = stablehlo.constant dense<2> : tensor<2x2xi64>
    %scaled = stablehlo.multiply %iota1, %c2 : tensor<2x2xi64>
    %added = stablehlo.add %scaled, %iota2 : tensor<2x2xi64>
    %c_offset = stablehlo.constant dense<1> : tensor<2x2xi64>
    %indices_2d = stablehlo.add %added, %c_offset : tensor<2x2xi64>
    // indices_2d is [[1, 2],
    //                [3, 4]]
    %indices = stablehlo.reshape %indices_2d : (tensor<2x2xi64>) -> tensor<4x1xi64>
    
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
// CHECK-LABEL: func.func @gather_reshaped_iota_to_slice
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %arg0 [1:5]
// CHECK-NEXT: return %[[SLICE]]

// Gather with 3D reshaped iota indices
func.func @gather_reshaped_3d_iota_to_slice(%arg0: tensor<100xi64>) -> tensor<2x2x2xi64> {
    %iota1 = stablehlo.iota dim = 0 : tensor<2x2x2xi64>
    %iota2 = stablehlo.iota dim = 1 : tensor<2x2x2xi64>
    %iota3 = stablehlo.iota dim = 2 : tensor<2x2x2xi64>
    %c10 = stablehlo.constant dense<10> : tensor<2x2x2xi64>
    %c5 = stablehlo.constant dense<5> : tensor<2x2x2xi64>
    %c_offset = stablehlo.constant dense<1> : tensor<2x2x2xi64>

    %scaled1 = stablehlo.multiply %iota1, %c10 : tensor<2x2x2xi64>
    %scaled2 = stablehlo.multiply %iota2, %c5 : tensor<2x2x2xi64>

    %added1 = stablehlo.add %scaled1, %scaled2 : tensor<2x2x2xi64>
    %added2 = stablehlo.add %added1, %iota3 : tensor<2x2x2xi64>
    %indices_3d = stablehlo.add %added2, %c_offset : tensor<2x2x2xi64>

    %indices = stablehlo.reshape %indices_3d : (tensor<2x2x2xi64>) -> tensor<8x1xi64>

    %0 = "stablehlo.gather"(%arg0, %indices) {
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1>
    } : (tensor<100xi64>, tensor<8x1xi64>) -> tensor<8xi64>

    %1 = stablehlo.reshape %0 : (tensor<8xi64>) -> tensor<2x2x2xi64>
    return %1 : tensor<2x2x2xi64>
}
// CHECK-LABEL: func.func @gather_reshaped_3d_iota_to_slice
// CHECK-NEXT: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<100xi64>) -> tensor<10x2x5xi64>
// CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %[[RESHAPE]] [0:2, 0:2, 1:3] : (tensor<10x2x5xi64>) -> tensor<2x2x2xi64>
// CHECK-NEXT: return %[[SLICE]]

func.func @main() {
    %input = stablehlo.constant dense<[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]> : tensor<10xi64>

    // 1. @gather_iota_to_slice
    %res1 = func.call @gather_iota_to_slice(%input) : (tensor<10xi64>) -> tensor<5xi64>
    %exp1 = stablehlo.constant dense<[10, 11, 12, 13, 14]> : tensor<5xi64>
    "check.expect_eq"(%res1, %exp1) : (tensor<5xi64>, tensor<5xi64>) -> ()

    // 2. @gather_iota_offset_to_slice
    %res2 = func.call @gather_iota_offset_to_slice(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp2 = stablehlo.constant dense<[12, 13, 14, 15]> : tensor<4xi64>
    "check.expect_eq"(%res2, %exp2) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 3. @gather_iota_stride_to_slice
    %res3 = func.call @gather_iota_stride_to_slice(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp3 = stablehlo.constant dense<[10, 12, 14, 16]> : tensor<4xi64>
    "check.expect_eq"(%res3, %exp3) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 4. @gather_iota_offset_stride_to_slice
    %res4 = func.call @gather_iota_offset_stride_to_slice(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp4 = stablehlo.constant dense<[11, 13, 15, 17]> : tensor<4xi64>
    "check.expect_eq"(%res4, %exp4) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 5. @gather_const_iota_to_slice
    %res5 = func.call @gather_const_iota_to_slice(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp5 = stablehlo.constant dense<[10, 11, 12, 13]> : tensor<4xi64>
    "check.expect_eq"(%res5, %exp5) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 6. @gather_scalar_index_to_dynamic_slice
    %idx = stablehlo.constant dense<[3]> : tensor<1xi64>
    %res6 = func.call @gather_scalar_index_to_dynamic_slice(%input, %idx) : (tensor<10xi64>, tensor<1xi64>) -> tensor<1xi64>
    %exp6 = stablehlo.constant dense<[13]> : tensor<1xi64>
    "check.expect_eq"(%res6, %exp6) : (tensor<1xi64>, tensor<1xi64>) -> ()

    // 7. @gather_iota_float
    %input_f = stablehlo.constant dense<[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]> : tensor<10xf64>
    %res7 = func.call @gather_iota_float(%input_f) : (tensor<10xf64>) -> tensor<5xf64>
    %exp7 = stablehlo.constant dense<[10.0, 11.0, 12.0, 13.0, 14.0]> : tensor<5xf64>
    "check.expect_eq"(%res7, %exp7) : (tensor<5xf64>, tensor<5xf64>) -> ()

    // 8. @gather_non_1d_operand
    %input_2d = stablehlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
    %res8 = func.call @gather_non_1d_operand(%input_2d) : (tensor<4x4xi64>) -> tensor<2xi64>
    %exp8 = stablehlo.constant dense<[0, 5]> : tensor<2xi64>
    "check.expect_eq"(%res8, %exp8) : (tensor<2xi64>, tensor<2xi64>) -> ()

    // 9. @gather_slice_size_not_1
    %res9 = func.call @gather_slice_size_not_1(%input) : (tensor<10xi64>) -> tensor<4x2xi64>
    %exp9 = stablehlo.constant dense<[[10, 11], [11, 12], [12, 13], [13, 14]]> : tensor<4x2xi64>
    "check.expect_eq"(%res9, %exp9) : (tensor<4x2xi64>, tensor<4x2xi64>) -> ()

    // 10. @gather_non_iota_indices
    %res10 = func.call @gather_non_iota_indices(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp10 = stablehlo.constant dense<[13, 11, 14, 12]> : tensor<4xi64>
    "check.expect_eq"(%res10, %exp10) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 11. @gather_iota_negative_stride_to_slice_reverse
    %res11 = func.call @gather_iota_negative_stride_to_slice_reverse(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp11 = stablehlo.constant dense<[14, 13, 12, 11]> : tensor<4xi64>
    "check.expect_eq"(%res11, %exp11) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 12. @gather_iota_negative_stride_offset_to_slice_reverse
    %res12 = func.call @gather_iota_negative_stride_offset_to_slice_reverse(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp12 = stablehlo.constant dense<[17, 15, 13, 11]> : tensor<4xi64>
    "check.expect_eq"(%res12, %exp12) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 13. @gather_reshaped_iota_to_slice
    %res13 = func.call @gather_reshaped_iota_to_slice(%input) : (tensor<10xi64>) -> tensor<4xi64>
    %exp13 = stablehlo.constant dense<[11, 12, 13, 14]> : tensor<4xi64>
    "check.expect_eq"(%res13, %exp13) : (tensor<4xi64>, tensor<4xi64>) -> ()

    // 14. @gather_reshaped_3d_iota_to_slice
    %input_100 = stablehlo.iota dim = 0 : tensor<100xi64>
    %res14 = func.call @gather_reshaped_3d_iota_to_slice(%input_100) : (tensor<100xi64>) -> tensor<2x2x2xi64>
    %exp14 = stablehlo.constant dense<[[[1, 2], [6, 7]], [[11, 12], [16, 17]]]> : tensor<2x2x2xi64>
    "check.expect_eq"(%res14, %exp14) : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> ()

    return
}
