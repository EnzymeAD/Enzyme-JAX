// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s

module {
  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @explicit_jacobian(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4x4xf64> {
    %j = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
    return %j : tensor<4x4xf64>
  }

  func.func @jactimes(%y: tensor<4xf64>, %yp: tensor<4xf64>, %v: tensor<4xf64>) -> tensor<4xf64> {
    return %v : tensor<4xf64>
  }

  func.func @explicit_sparse_direct(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @explicit_jacobian
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) {effective_jacobian = "dF/dy + cj*dF/dyp"} :
      (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }

  func.func @jacobian_action_iterative(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian_action = @jactimes
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%y, %yp) {effective_jacobian = "dF/dy + cj*dF/dyp"} :
      (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }
}

// CHECK-LABEL: func.func @explicit_sparse_direct
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @explicit_jacobian
// CHECK-SAME: linear_solver = <explicit_sparse_direct>
// CHECK-SAME: jacobian_demand = <explicit_matrix>
// CHECK-SAME: effective_jacobian = "dF/dy + cj*dF/dyp"

// CHECK-LABEL: func.func @jacobian_action_iterative
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian_action = @jactimes
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: effective_jacobian = "dF/dy + cj*dF/dyp"

// -----

module {
  func.func @materialize_sparse_jacobian() {
    return
  }

  func.func @residual_wrapper() {
    return
  }

  enzymexla.jacobian_materialization materializer = @materialize_sparse_jacobian
    residual = @residual_wrapper
    method = <one_hot_forward>
    storage = <sparse_callback>
    fwddiff_calls = 4
    todense_calls = 8
    sparse_store_callbacks = 4
    {active_input_dimension_arg = 2 : i64,
     active_input_index = 1 : i64,
     active_input_index_map_arg = 4 : i64,
     active_output_index = 0 : i64,
     enzyme_activity = ["enzyme_const", "enzyme_dup", "enzyme_dupnoneed"],
     input_activity = ["enzyme_const", "enzyme_dup"],
     input_count = 2 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_dimension_arg = 1 : i64,
     output_index_map_arg = 3 : i64,
     output_count = 1 : i64,
     seed_loop_dimension_arg = 2 : i64,
     source = "unit-test",
     sparse_assembly = "coo_column_seeded_callback",
     sparse_cols_arg = 9 : i64,
     sparse_nnz_arg = 11 : i64,
     sparse_rows_arg = 8 : i64,
     sparse_values_arg = 10 : i64}
}

// CHECK: enzymexla.jacobian_materialization materializer = @materialize_sparse_jacobian
// CHECK-SAME: residual = @residual_wrapper
// CHECK-SAME: method = <one_hot_forward>
// CHECK-SAME: storage = <sparse_callback>
// CHECK-SAME: fwddiff_calls = 4
// CHECK-SAME: todense_calls = 8
// CHECK-SAME: sparse_store_callbacks = 4
// CHECK-SAME: active_input_dimension_arg = 2 : i64
// CHECK-SAME: active_input_index = 1 : i64
// CHECK-SAME: active_input_index_map_arg = 4 : i64
// CHECK-SAME: active_output_index = 0 : i64
// CHECK-SAME: enzyme_activity = ["enzyme_const", "enzyme_dup", "enzyme_dupnoneed"]
// CHECK-SAME: input_activity = ["enzyme_const", "enzyme_dup"]
// CHECK-SAME: input_count = 2 : i64
// CHECK-SAME: output_activity = ["enzyme_dupnoneed"]
// CHECK-SAME: output_count = 1 : i64
// CHECK-SAME: output_dimension_arg = 1 : i64
// CHECK-SAME: output_index_map_arg = 3 : i64
// CHECK-SAME: seed_loop_dimension_arg = 2 : i64
// CHECK-SAME: source = "unit-test"
// CHECK-SAME: sparse_assembly = "coo_column_seeded_callback"
// CHECK-SAME: sparse_cols_arg = 9 : i64
// CHECK-SAME: sparse_nnz_arg = 11 : i64
// CHECK-SAME: sparse_rows_arg = 8 : i64
// CHECK-SAME: sparse_values_arg = 10 : i64
