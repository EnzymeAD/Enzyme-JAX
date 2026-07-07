// RUN: enzymexlamlir-opt --select-sundials-ida-matrix-free %s | FileCheck %s

module {
  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @dfdy_materializer() {
    return
  }

  func.func @dfdyp_materializer() {
    return
  }

  enzymexla.jacobian_action @plain_dfdy_action
    materialization = @dfdy_materializer
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {source = "unit-test"}

  enzymexla.jacobian_action @ida_effective_action
    materialization = @dfdy_materializer
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {enzymexla.sundials.ida_effective_jacobian_action,
     source = "unit-test",
     yp_active_input_index = 2 : i64,
     yp_materialization = @dfdyp_materializer}

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

// CHECK: module attributes {enzymexla.sundials.ida_matrix_free_selected = 1 : i64}
// CHECK-LABEL: func.func @solve
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @dfdy_materializer
// CHECK-SAME: jacobian_action = @ida_effective_action
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
