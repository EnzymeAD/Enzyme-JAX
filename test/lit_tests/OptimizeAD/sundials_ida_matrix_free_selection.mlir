// RUN: enzymexlamlir-opt --synthesize-sundials-ida-jacobian-actions --select-sundials-ida-matrix-free %s | FileCheck %s

module {
  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @other_residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %yp : tensor<4xf64>
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
     active_output_index = 0 : i64,
     input_activity = ["enzyme_const", "enzyme_dup"],
     input_count = 2 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_count = 1 : i64,
     output_dimension_arg = 1 : i64,
     seed_loop_dimension_arg = 2 : i64,
     sparse_assembly = "coo_column_seeded_callback"}

  func.func @direct_opt_in(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @materialize_sparse_jacobian
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) {enzymexla.sundials.allow_matrix_free}
      : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }

  func.func @direct_residual_match(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) {enzymexla.sundials.allow_matrix_free}
      : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }

  func.func @direct_no_opt_in(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @materialize_sparse_jacobian
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }

  func.func @direct_no_action(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @other_residual
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) {enzymexla.sundials.allow_matrix_free}
      : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.jacobian_actions_synthesized = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_matrix_free_selected = 2 : i64

// CHECK: enzymexla.jacobian_action @__enzymexla_jacobian_action_0 materialization = @materialize_sparse_jacobian residual = @residual

// CHECK-LABEL: func.func @direct_opt_in
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @materialize_sparse_jacobian
// CHECK-SAME: jacobian_action = @__enzymexla_jacobian_action_0
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: enzymexla.sundials.allow_matrix_free
// CHECK-SAME: enzymexla.sundials.matrix_free_selected

// CHECK-LABEL: func.func @direct_residual_match
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian_action = @__enzymexla_jacobian_action_0
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>

// CHECK-LABEL: func.func @direct_no_opt_in
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @materialize_sparse_jacobian
// CHECK-SAME: linear_solver = <explicit_sparse_direct>
// CHECK-SAME: jacobian_demand = <explicit_matrix>
// CHECK-NOT: jacobian_action =

// CHECK-LABEL: func.func @direct_no_action
// CHECK: enzymexla.sundials.ida_solve residual = @other_residual
// CHECK-SAME: linear_solver = <explicit_sparse_direct>
// CHECK-SAME: jacobian_demand = <explicit_matrix>
// CHECK-NOT: jacobian_action =
