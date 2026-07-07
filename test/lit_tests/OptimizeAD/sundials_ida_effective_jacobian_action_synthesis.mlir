// RUN: enzymexlamlir-opt --synthesize-sundials-ida-jacobian-actions --select-sundials-ida-matrix-free %s | FileCheck %s

module {
  func.func @residual(%model: !llvm.ptr, %y: tensor<4xf64>, %yp: tensor<4xf64>, %out: tensor<4xf64>) -> tensor<4xf64> {
    return %out : tensor<4xf64>
  }

  func.func @dfdy_materializer() {
    return
  }

  func.func @dfdyp_materializer() {
    return
  }

  enzymexla.jacobian_materialization materializer = @dfdy_materializer
    residual = @residual
    method = <one_hot_forward>
    storage = <sparse_callback>
    fwddiff_calls = 4
    todense_calls = 8
    sparse_store_callbacks = 4
    {active_input_dimension_arg = 2 : i64,
     active_input_index = 1 : i64,
     active_input_index_map_arg = 4 : i64,
     active_output_index = 0 : i64,
     input_activity = ["enzyme_const", "enzyme_dup", "enzyme_const"],
     input_count = 3 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_count = 1 : i64,
     output_dimension_arg = 1 : i64,
     output_index_map_arg = 3 : i64,
     seed_loop_dimension_arg = 2 : i64,
     source = "DfDy",
     sparse_assembly = "coo_column_seeded_callback",
     sparse_cols_arg = 9 : i64,
     sparse_nnz_arg = 11 : i64,
     sparse_rows_arg = 8 : i64,
     sparse_values_arg = 10 : i64}

  enzymexla.jacobian_materialization materializer = @dfdyp_materializer
    residual = @residual
    method = <one_hot_forward>
    storage = <sparse_callback>
    fwddiff_calls = 4
    todense_calls = 8
    sparse_store_callbacks = 4
    {active_input_dimension_arg = 2 : i64,
     active_input_index = 2 : i64,
     active_input_index_map_arg = 4 : i64,
     active_output_index = 0 : i64,
     input_activity = ["enzyme_const", "enzyme_const", "enzyme_dup"],
     input_count = 3 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_count = 1 : i64,
     output_dimension_arg = 1 : i64,
     output_index_map_arg = 3 : i64,
     seed_loop_dimension_arg = 2 : i64,
     source = "DfDyp",
     sparse_assembly = "coo_column_seeded_callback",
     sparse_cols_arg = 10 : i64,
     sparse_nnz_arg = 12 : i64,
     sparse_rows_arg = 9 : i64,
     sparse_values_arg = 11 : i64}

  func.func @solve(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @dfdy_materializer
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) {enzymexla.sundials.allow_matrix_free}
      : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.jacobian_actions_synthesized = 3 : i64
// CHECK-SAME: enzymexla.sundials.ida_effective_jacobian_actions_synthesized = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_matrix_free_selected = 1 : i64

// CHECK: enzymexla.jacobian_action @__enzymexla_sundials_ida_effective_jacobian_action_0 materialization = @dfdy_materializer residual = @residual active_input_index = 1 active_output_index = 0
// CHECK-SAME: enzymexla.sundials.ida_effective_jacobian_action
// CHECK-SAME: y_materialization = @dfdy_materializer
// CHECK-SAME: yp_active_input_index = 2 : i64
// CHECK-SAME: yp_materialization = @dfdyp_materializer

// CHECK-LABEL: func.func @solve
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @dfdy_materializer
// CHECK-SAME: jacobian_action = @__enzymexla_sundials_ida_effective_jacobian_action_0
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: enzymexla.sundials.matrix_free_selected
