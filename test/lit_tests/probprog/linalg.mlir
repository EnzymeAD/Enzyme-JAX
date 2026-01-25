// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // Test: enzyme.cholesky with lower=true (default)
  // CHECK:       func.func @test_cholesky_lower(%[[ARG0:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.cholesky %[[ARG0]], lower = true : tensor<3x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x3xf64>
  // CHECK-NEXT:  }
  func.func @test_cholesky_lower(%input: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %result = enzyme.cholesky %input {lower = true} : (tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
  }

  // Test: enzyme.cholesky with lower=false
  // CHECK:       func.func @test_cholesky_upper(%[[ARG0:.+]]: tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.cholesky %[[ARG0]] : tensor<4x4xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf64>
  // CHECK-NEXT:  }
  func.func @test_cholesky_upper(%input: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %result = enzyme.cholesky %input {lower = false} : (tensor<4x4xf64>) -> tensor<4x4xf64>
    return %result : tensor<4x4xf64>
  }

  // Test: enzyme.triangular_solve with left_side=true, lower=true (defaults)
  // CHECK:       func.func @test_triangular_solve_basic(%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x3xf64>
  // CHECK-NEXT:  }
  func.func @test_triangular_solve_basic(%a: tensor<3x3xf64>, %b: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %result = enzyme.triangular_solve %a, %b {left_side = true, lower = true, unit_diagonal = false, transpose_a = #enzyme<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
  }

  // Test: enzyme.triangular_solve with left_side=false
  // CHECK:       func.func @test_triangular_solve_right(%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x3xf64>
  // CHECK-NEXT:  }
  func.func @test_triangular_solve_right(%a: tensor<3x3xf64>, %b: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %result = enzyme.triangular_solve %a, %b {left_side = false, lower = true, unit_diagonal = false, transpose_a = #enzyme<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
  }

  // Test: enzyme.triangular_solve with lower=false (upper triangular)
  // CHECK:       func.func @test_triangular_solve_upper(%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x3xf64>
  // CHECK-NEXT:  }
  func.func @test_triangular_solve_upper(%a: tensor<3x3xf64>, %b: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %result = enzyme.triangular_solve %a, %b {left_side = true, lower = false, unit_diagonal = false, transpose_a = #enzyme<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
  }

  // Test: enzyme.triangular_solve with unit_diagonal=true
  // CHECK:       func.func @test_triangular_solve_unit_diag(%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x3xf64>
  // CHECK-NEXT:  }
  func.func @test_triangular_solve_unit_diag(%a: tensor<3x3xf64>, %b: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %result = enzyme.triangular_solve %a, %b {left_side = true, lower = true, unit_diagonal = true, transpose_a = #enzyme<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
  }

  // Test: enzyme.triangular_solve with transpose_a=TRANSPOSE
  // CHECK:       func.func @test_triangular_solve_transpose(%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x3xf64>
  // CHECK-NEXT:  }
  func.func @test_triangular_solve_transpose(%a: tensor<3x3xf64>, %b: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %result = enzyme.triangular_solve %a, %b {left_side = true, lower = true, unit_diagonal = false, transpose_a = #enzyme<transpose TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
  }

  // Test: enzyme.triangular_solve with 1D b (vector)
  // The conversion reshapes 1D b to 2D column matrix, solves, then reshapes back
  // CHECK:       func.func @test_triangular_solve_vector(%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3xf64>) -> tensor<3xf64> {
  // CHECK-NEXT:    %[[RESHAPED:.+]] = stablehlo.reshape %[[B]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT:    %[[SOLVED:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[RESHAPED]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.reshape %[[SOLVED]] : (tensor<3x1xf64>) -> tensor<3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3xf64>
  // CHECK-NEXT:  }
  func.func @test_triangular_solve_vector(%a: tensor<3x3xf64>, %b: tensor<3xf64>) -> tensor<3xf64> {
    %result = enzyme.triangular_solve %a, %b {left_side = true, lower = true, unit_diagonal = false, transpose_a = #enzyme<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %result : tensor<3xf64>
  }

  // Test: enzyme.dot (matrix-vector multiplication)
  // CHECK:       func.func @test_dot_matvec(%[[LHS:.+]]: tensor<3x4xf64>, %[[RHS:.+]]: tensor<4xf64>) -> tensor<3xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]], contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4xf64>) -> tensor<3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3xf64>
  // CHECK-NEXT:  }
  func.func @test_dot_matvec(%lhs: tensor<3x4xf64>, %rhs: tensor<4xf64>) -> tensor<3xf64> {
    %result = enzyme.dot %lhs, %rhs {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<3x4xf64>, tensor<4xf64>) -> tensor<3xf64>
    return %result : tensor<3xf64>
  }

  // Test: enzyme.dot (matrix-matrix multiplication)
  // CHECK:       func.func @test_dot_matmul(%[[LHS:.+]]: tensor<3x4xf64>, %[[RHS:.+]]: tensor<4x5xf64>) -> tensor<3x5xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]], contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4x5xf64>) -> tensor<3x5xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x5xf64>
  // CHECK-NEXT:  }
  func.func @test_dot_matmul(%lhs: tensor<3x4xf64>, %rhs: tensor<4x5xf64>) -> tensor<3x5xf64> {
    %result = enzyme.dot %lhs, %rhs {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<3x4xf64>, tensor<4x5xf64>) -> tensor<3x5xf64>
    return %result : tensor<3x5xf64>
  }

  // Test: enzyme.dot (vector-vector inner product)
  // CHECK:       func.func @test_dot_inner(%[[LHS:.+]]: tensor<4xf64>, %[[RHS:.+]]: tensor<4xf64>) -> tensor<f64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]], contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<f64>
  // CHECK-NEXT:  }
  func.func @test_dot_inner(%lhs: tensor<4xf64>, %rhs: tensor<4xf64>) -> tensor<f64> {
    %result = enzyme.dot %lhs, %rhs {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // Test: enzyme.dot with batching dimensions
  // CHECK:       func.func @test_dot_batched(%[[LHS:.+]]: tensor<2x3x4xf64>, %[[RHS:.+]]: tensor<2x4x5xf64>) -> tensor<2x3x5xf64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xf64>, tensor<2x4x5xf64>) -> tensor<2x3x5xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<2x3x5xf64>
  // CHECK-NEXT:  }
  func.func @test_dot_batched(%lhs: tensor<2x3x4xf64>, %rhs: tensor<2x4x5xf64>) -> tensor<2x3x5xf64> {
    %result = enzyme.dot %lhs, %rhs {lhs_batching_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64: 0>, lhs_contracting_dimensions = array<i64: 2>, rhs_contracting_dimensions = array<i64: 1>} : (tensor<2x3x4xf64>, tensor<2x4x5xf64>) -> tensor<2x3x5xf64>
    return %result : tensor<2x3x5xf64>
  }
}
