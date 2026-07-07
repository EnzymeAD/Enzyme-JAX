// RUN: enzymexlamlir-opt --synthesize-sundials-ida-jacobian-actions %s | FileCheck %s

module {
  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @materialize_sparse_jacobian(%model: !llvm.ptr, %n_res: i64, %n_var: i64) {
    return
  }

  enzymexla.jacobian_materialization materializer = @materialize_sparse_jacobian
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
     input_activity = ["enzyme_const", "enzyme_dup"],
     input_count = 2 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_count = 1 : i64,
     output_dimension_arg = 1 : i64,
     output_index_map_arg = 3 : i64,
     seed_loop_dimension_arg = 2 : i64,
     source = "unit-test",
     sparse_assembly = "coo_column_seeded_callback",
     sparse_cols_arg = 9 : i64,
     sparse_nnz_arg = 11 : i64,
     sparse_rows_arg = 8 : i64,
     sparse_values_arg = 10 : i64}

  func.func @action_solve(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @materialize_sparse_jacobian
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%y, %yp) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }

  func.func @direct_solve(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @materialize_sparse_jacobian
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }
}

// CHECK: module attributes {enzymexla.jacobian_actions_synthesized = 1 : i64, enzymexla.sundials.ida_solves_linked_jacobian_actions = 1 : i64}
// CHECK: enzymexla.jacobian_action @__enzymexla_jacobian_action_0 materialization = @materialize_sparse_jacobian residual = @residual active_input_index = 1 active_output_index = 0
// CHECK-SAME: active_input_dimension_arg = 2 : i64
// CHECK-SAME: active_input_index_map_arg = 4 : i64
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

// CHECK-LABEL: func.func @action_solve
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @materialize_sparse_jacobian
// CHECK-SAME: jacobian_action = @__enzymexla_jacobian_action_0
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>

// CHECK-LABEL: func.func @direct_solve
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @materialize_sparse_jacobian
// CHECK-SAME: linear_solver = <explicit_sparse_direct>
// CHECK-SAME: jacobian_demand = <explicit_matrix>
// CHECK-NOT: jacobian_action =
