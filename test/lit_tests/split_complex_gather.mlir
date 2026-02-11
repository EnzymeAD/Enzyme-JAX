// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test: Basic gather on complex tensors
// CHECK-LABEL: @test_basic_gather
func.func @test_basic_gather(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>) -> tensor<2xcomplex<f32>> {
  %0 = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1>,
    indices_are_sorted = false
  } : (tensor<4xcomplex<f32>>, tensor<2x1xi64>) -> tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
}
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_GATHER:.*]] = "stablehlo.gather"(%[[REAL_INPUT]], %arg1)
// CHECK-SAME: dimension_numbers = #stablehlo.gather<
// CHECK-SAME:   collapsed_slice_dims = [0],
// CHECK-SAME:   start_index_map = [0],
// CHECK-SAME:   index_vector_dim = 1
// CHECK-SAME: >,
// CHECK-SAME: slice_sizes = array<i64: 1>
// CHECK-SAME: (tensor<4xf32>, tensor<2x1xi64>) -> tensor<2xf32>
// CHECK: %[[IMAG_GATHER:.*]] = "stablehlo.gather"(%[[IMAG_INPUT]], %arg1)
// CHECK-SAME: dimension_numbers = #stablehlo.gather<
// CHECK-SAME:   collapsed_slice_dims = [0],
// CHECK-SAME:   start_index_map = [0],
// CHECK-SAME:   index_vector_dim = 1
// CHECK-SAME: >,
// CHECK-SAME: slice_sizes = array<i64: 1>
// CHECK-SAME: (tensor<4xf32>, tensor<2x1xi64>) -> tensor<2xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_GATHER]], %[[IMAG_GATHER]]
// CHECK: return %[[RESULT]]

// Test: Multi-dimensional gather on complex tensors
// CHECK-LABEL: @test_multidim_gather
func.func @test_multidim_gather(%input: tensor<3x4xcomplex<f32>>, %indices: tensor<2x1xi64>) -> tensor<2x4xcomplex<f32>> {
  %0 = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1, 4>,
    indices_are_sorted = false
  } : (tensor<3x4xcomplex<f32>>, tensor<2x1xi64>) -> tensor<2x4xcomplex<f32>>
  return %0 : tensor<2x4xcomplex<f32>>
}
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_GATHER:.*]] = "stablehlo.gather"(%[[REAL_INPUT]], %arg1)
// CHECK-SAME: dimension_numbers = #stablehlo.gather<
// CHECK-SAME:   offset_dims = [1],
// CHECK-SAME:   collapsed_slice_dims = [0],
// CHECK-SAME:   start_index_map = [0],
// CHECK-SAME:   index_vector_dim = 1
// CHECK-SAME: >,
// CHECK-SAME: slice_sizes = array<i64: 1, 4>
// CHECK-SAME: (tensor<3x4xf32>, tensor<2x1xi64>) -> tensor<2x4xf32>
// CHECK: %[[IMAG_GATHER:.*]] = "stablehlo.gather"(%[[IMAG_INPUT]], %arg1)
// CHECK-SAME: dimension_numbers = #stablehlo.gather<
// CHECK-SAME:   offset_dims = [1],
// CHECK-SAME:   collapsed_slice_dims = [0],
// CHECK-SAME:   start_index_map = [0],
// CHECK-SAME:   index_vector_dim = 1
// CHECK-SAME: >,
// CHECK-SAME: slice_sizes = array<i64: 1, 4>
// CHECK-SAME: (tensor<3x4xf32>, tensor<2x1xi64>) -> tensor<2x4xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_GATHER]], %[[IMAG_GATHER]]
// CHECK: return %[[RESULT]]

// Test: Complex gather with complex<f64> type
// CHECK-LABEL: @test_complex_f64
func.func @test_complex_f64(%input: tensor<8xcomplex<f64>>, %indices: tensor<3x1xi64>) -> tensor<3xcomplex<f64>> {
  %0 = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1>,
    indices_are_sorted = false
  } : (tensor<8xcomplex<f64>>, tensor<3x1xi64>) -> tensor<3xcomplex<f64>>
  return %0 : tensor<3xcomplex<f64>>
}
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_GATHER:.*]] = "stablehlo.gather"(%[[REAL_INPUT]], %arg1)
// CHECK-SAME: (tensor<8xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK: %[[IMAG_GATHER:.*]] = "stablehlo.gather"(%[[IMAG_INPUT]], %arg1)
// CHECK-SAME: (tensor<8xf64>, tensor<3x1xi64>) -> tensor<3xf64>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_GATHER]], %[[IMAG_GATHER]]
// CHECK: return %[[RESULT]]

// Test: Non-complex gather should not be transformed
// CHECK-LABEL: @test_non_complex
func.func @test_non_complex(%input: tensor<4xf32>, %indices: tensor<2x1xi64>) -> tensor<2xf32> {
  %0 = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1>,
    indices_are_sorted = false
  } : (tensor<4xf32>, tensor<2x1xi64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-NOT: stablehlo.real
// CHECK-NOT: stablehlo.imag
// CHECK-NOT: stablehlo.complex
// CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %arg1)
// CHECK: return %[[RESULT]]
