// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

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
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_SCATTER:.*]] = "stablehlo.scatter"(%[[REAL_INPUT]], %arg1, %[[REAL_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG1]] : tensor<f32>
// CHECK: %[[IMAG_SCATTER:.*]] = "stablehlo.scatter"(%[[IMAG_INPUT]], %arg1, %[[IMAG_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG1]] : tensor<f32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_SCATTER]], %[[IMAG_SCATTER]]
// CHECK: return %[[RESULT]]

// Test: ConstantSetindex scatter on complex tensors
func.func @test_constant_setindex(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(3.0, 4.0)> : tensor<complex<f32>>
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    stablehlo.return %cst : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK-LABEL: @test_constant_setindex
// CHECK-DAG: %[[IMAG_CST:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK-DAG: %[[REAL_CST:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_SCATTER:.*]] = "stablehlo.scatter"(%[[REAL_INPUT]], %arg1, %[[REAL_UPDATE]])
// CHECK: ^bb0(%{{.*}}: tensor<f32>, %{{.*}}: tensor<f32>):
// CHECK:   stablehlo.return %[[REAL_CST]] : tensor<f32>
// CHECK: %[[IMAG_SCATTER:.*]] = "stablehlo.scatter"(%[[IMAG_INPUT]], %arg1, %[[IMAG_UPDATE]])
// CHECK: ^bb0(%{{.*}}: tensor<f32>, %{{.*}}: tensor<f32>):
// CHECK:   stablehlo.return %[[IMAG_CST]] : tensor<f32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_SCATTER]], %[[IMAG_SCATTER]]
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
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_SCATTER:.*]] = "stablehlo.scatter"(%[[REAL_INPUT]], %arg1, %[[REAL_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: %[[IMAG_SCATTER:.*]] = "stablehlo.scatter"(%[[IMAG_INPUT]], %arg1, %[[IMAG_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_SCATTER]], %[[IMAG_SCATTER]]
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
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_SCATTER:.*]] = "stablehlo.scatter"(%[[REAL_INPUT]], %arg1, %[[REAL_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[SUB:.*]] = stablehlo.subtract %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[SUB]] : tensor<f32>
// CHECK: %[[IMAG_SCATTER:.*]] = "stablehlo.scatter"(%[[IMAG_INPUT]], %arg1, %[[IMAG_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[SUB:.*]] = stablehlo.subtract %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[SUB]] : tensor<f32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_SCATTER]], %[[IMAG_SCATTER]]
// CHECK: return %[[RESULT]]

// Test: AddConstantUpdate scatter on complex tensors
func.func @test_add_constant_update(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(2.0, 5.0)> : tensor<complex<f32>>
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %arg0, %cst : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK-LABEL: @test_add_constant_update
// CHECK-DAG: %[[IMAG_CST:.*]] = stablehlo.constant dense<5.000000e+00> : tensor<f32>
// CHECK-DAG: %[[REAL_CST:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_SCATTER:.*]] = "stablehlo.scatter"(%[[REAL_INPUT]], %arg1, %[[REAL_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %{{.*}}: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[REAL_CST]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: %[[IMAG_SCATTER:.*]] = "stablehlo.scatter"(%[[IMAG_INPUT]], %arg1, %[[IMAG_UPDATE]])
// CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %{{.*}}: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[IMAG_CST]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_SCATTER]], %[[IMAG_SCATTER]]
// CHECK: return %[[RESULT]]

// Test: AddConstantInput scatter on complex tensors
func.func @test_add_constant_input(%input: tensor<4xcomplex<f32>>, %indices: tensor<2x1xi64>, %updates: tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(1.5, 2.5)> : tensor<complex<f32>>
  %0 = "stablehlo.scatter"(%input, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    %1 = stablehlo.add %cst, %arg1 : tensor<complex<f32>>
    stablehlo.return %1 : tensor<complex<f32>>
  }) : (tensor<4xcomplex<f32>>, tensor<2x1xi64>, tensor<2xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}
// CHECK-LABEL: @test_add_constant_input
// CHECK-DAG: %[[IMAG_CST:.*]] = stablehlo.constant dense<2.500000e+00> : tensor<f32>
// CHECK-DAG: %[[REAL_CST:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<f32>
// CHECK: %[[REAL_INPUT:.*]] = stablehlo.real %arg0
// CHECK: %[[IMAG_INPUT:.*]] = stablehlo.imag %arg0
// CHECK: %[[REAL_UPDATE:.*]] = stablehlo.real %arg2
// CHECK: %[[IMAG_UPDATE:.*]] = stablehlo.imag %arg2
// CHECK: %[[REAL_SCATTER:.*]] = "stablehlo.scatter"(%[[REAL_INPUT]], %arg1, %[[REAL_UPDATE]])
// CHECK: ^bb0(%{{.*}}: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[REAL_CST]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: %[[IMAG_SCATTER:.*]] = "stablehlo.scatter"(%[[IMAG_INPUT]], %arg1, %[[IMAG_UPDATE]])
// CHECK: ^bb0(%{{.*}}: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[IMAG_CST]], %[[ARG1]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: %[[RESULT:.*]] = stablehlo.complex %[[REAL_SCATTER]], %[[IMAG_SCATTER]]
// CHECK: return %[[RESULT]]
