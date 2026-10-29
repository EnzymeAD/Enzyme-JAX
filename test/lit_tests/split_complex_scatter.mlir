// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// The real/imag parts are packed along a new *trailing* dimension of size 2
// (f32[N,2] has the memory layout of complex<f32>[N]) so that XLA does not need
// to transpose the updates into its canonical scatter form; see
// https://github.com/EnzymeAD/Reactant.jl/issues/3563.

// Test: Setindex scatter on complex tensors
// CHECK-LABEL: @test_setindex
func.func @test_setindex(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    stablehlo.return %arg1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_INPUT_R:.*]] = stablehlo.reshape %[[REAL_INPUT]] : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK: %[[IMAG_INPUT_R:.*]] = stablehlo.reshape %[[IMAG_INPUT]] : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %[[REAL_INPUT_R]], %[[IMAG_INPUT_R]], dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_UPDATE_R:.*]] = stablehlo.reshape %[[REAL_UPDATE]] : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK: %[[IMAG_UPDATE_R:.*]] = stablehlo.reshape %[[IMAG_UPDATE]] : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK: %[[CONCAT_UPDATE:.*]] = stablehlo.concatenate %[[REAL_UPDATE_R]], %[[IMAG_UPDATE_R]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CONCAT_UPDATE]])
// CHECK-SAME: update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG1]] : tensor<f32>
// CHECK: (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>
// CHECK: %[[REAL_S:.*]] = stablehlo.slice %[[SCATTER]] [0:4, 0:1] : (tensor<4x2xf32>) -> tensor<4x1xf32>
// CHECK: %[[IMAG_S:.*]] = stablehlo.slice %[[SCATTER]] [0:4, 1:2] : (tensor<4x2xf32>) -> tensor<4x1xf32>
// CHECK: %[[REAL_RES:.*]] = stablehlo.reshape %[[REAL_S]] : (tensor<4x1xf32>) -> tensor<4xf32>
// CHECK: %[[IMAG_RES:.*]] = stablehlo.reshape %[[IMAG_S]] : (tensor<4x1xf32>) -> tensor<4xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_RES]], %[[IMAG_RES]]
// CHECK: return %[[RESULT]]

// Test: ConstantSetindex scatter on complex tensors
// CHECK-LABEL: @test_constant_setindex
func.func @test_constant_setindex(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(3.0, 4.0)> : tensor<complex<f32>>
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    stablehlo.return %cst : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// The constant real/imag parts get folded into a single constant tensor
// CHECK: %[[CST:.*]] = stablehlo.constant dense<{{\[}}[3.000000e+00, 4.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_INPUT_R:.*]] = stablehlo.reshape %[[REAL_INPUT]]
// CHECK: %[[IMAG_INPUT_R:.*]] = stablehlo.reshape %[[IMAG_INPUT]]
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %[[REAL_INPUT_R]], %[[IMAG_INPUT_R]], dim = 1
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CST]])
// CHECK-SAME: update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: ^bb0(%{{.*}}: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG1]] : tensor<f32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]

// Test: Add scatter on complex tensors
// CHECK-LABEL: @test_add
func.func @test_add(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
// CHECK: %[[CONCAT_UPDATE:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CONCAT_UPDATE]])
// CHECK-SAME: update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]

// Test: Sub scatter on complex tensors
// CHECK-LABEL: @test_sub
func.func @test_sub(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
// CHECK: %[[CONCAT_UPDATE:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CONCAT_UPDATE]])
// CHECK-SAME: update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[SUB:.*]] = stablehlo.subtract %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[SUB]] : tensor<f32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]

// Test: AddConstantUpdate scatter on complex tensors
// CHECK-LABEL: @test_add_constant_update
func.func @test_add_constant_update(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(2.0, 5.0)> : tensor<complex<f32>>
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %arg0, %cst : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK: %[[CST:.*]] = stablehlo.constant dense<{{\[}}[2.000000e+00, 5.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<2x2xf32>
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CST]])
// CHECK-SAME: update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]

// Test: AddConstantInput scatter on complex tensors
// CHECK-LABEL: @test_add_constant_input
func.func @test_add_constant_input(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(1.5, 2.5)> : tensor<complex<f32>>
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %cst, %arg1 : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// The constant + update gets precomputed and folded
// CHECK: %[[CST:.*]] = stablehlo.constant dense<{{\[}}[1.500000e+00, 2.500000e+00], [1.500000e+00, 2.500000e+00]]> : tensor<2x2xf32>
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
// CHECK: %[[CONCAT_UPDATE:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CST]], %[[CONCAT_UPDATE]]
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[ADD]])
// CHECK-SAME: update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: ^bb0(%{{.*}}: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG1]] : tensor<f32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:4, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]

// Test: Add scatter with pre-existing window dims; the new real/imag dim is
// appended to update_window_dims and all other dimension numbers are untouched.
// CHECK-LABEL: @test_add_window_dims
func.func @test_add_window_dims(%input: tensor<3x4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>> {
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<3x4xcomplex<f32>>, tensor<2x1xi64>, tensor<2x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
  return %0 : tensor<3x4xcomplex<f32>>
}
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 2 : (tensor<3x4x1xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
// CHECK: %[[CONCAT_UPDATE:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 2 : (tensor<2x4x1xf32>, tensor<2x4x1xf32>) -> tensor<2x4x2xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CONCAT_UPDATE]])
// CHECK-SAME: update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1
// CHECK: (tensor<3x4x2xf32>, tensor<2x1xi64>, tensor<2x4x2xf32>) -> tensor<3x4x2xf32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:3, 0:4, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:3, 0:4, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]

// Test: Add scatter with batching dims
// CHECK-LABEL: @test_add_batching
func.func @test_add_batching(%input: tensor<5x8xcomplex<f32>>, %indices: tensor<5x3x1xi64>, %updates: tensor<5x3xcomplex<f32>>) -> tensor<5x8xcomplex<f32>> {
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<5x8xcomplex<f32>>, tensor<5x3x1xi64>, tensor<5x3xcomplex<f32>>) -> tensor<5x8xcomplex<f32>>
  return %0 : tensor<5x8xcomplex<f32>>
}
// CHECK: %[[CONCAT_INPUT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 2 : (tensor<5x8x1xf32>, tensor<5x8x1xf32>) -> tensor<5x8x2xf32>
// CHECK: %[[CONCAT_UPDATE:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 2 : (tensor<5x3x1xf32>, tensor<5x3x1xf32>) -> tensor<5x3x2xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CONCAT_INPUT]], %arg1, %[[CONCAT_UPDATE]])
// CHECK-SAME: update_window_dims = [2], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2
// CHECK: (tensor<5x8x2xf32>, tensor<5x3x1xi64>, tensor<5x3x2xf32>) -> tensor<5x8x2xf32>
// CHECK: stablehlo.slice %[[SCATTER]] [0:5, 0:8, 0:1]
// CHECK: stablehlo.slice %[[SCATTER]] [0:5, 0:8, 1:2]
// CHECK: %[[RESULT:.*]] = stablehlo.complex
// CHECK: return %[[RESULT]]
