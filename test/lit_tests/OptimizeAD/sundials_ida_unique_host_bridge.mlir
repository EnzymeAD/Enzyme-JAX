// RUN: enzymexlamlir-opt --synthesize-sundials-ida-jacobian-actions --select-sundials-ida-matrix-free %s | FileCheck %s --check-prefix=NO-BRIDGE
// RUN: enzymexlamlir-opt --synthesize-sundials-ida-jacobian-actions='allow-unique-host-jacobian-bridge=true' --select-sundials-ida-matrix-free %s | FileCheck %s --check-prefix=BRIDGE

module {
  func.func @component_residual(%model: !llvm.ptr, %y: tensor<4xf64>, %yp: tensor<4xf64>, %out: tensor<4xf64>) -> tensor<4xf64> {
    return %out : tensor<4xf64>
  }

  func.func @dfdy_materializer() {
    return
  }

  func.func @dfdyp_materializer() {
    return
  }

  func.func @ida_residual() {
    return
  }

  func.func @ida_jac_callback() {
    return
  }

  enzymexla.jacobian_materialization materializer = @dfdy_materializer
    residual = @component_residual
    method = <one_hot_forward>
    storage = <sparse_callback>
    fwddiff_calls = 4
    todense_calls = 8
    sparse_store_callbacks = 4
    {active_input_dimension_arg = 2 : i64,
     active_input_index = 1 : i64,
     active_output_index = 0 : i64,
     input_activity = ["enzyme_const", "enzyme_dup", "enzyme_const"],
     input_count = 3 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_count = 1 : i64,
     output_dimension_arg = 1 : i64,
     seed_loop_dimension_arg = 2 : i64,
     source = "DfDy",
     sparse_assembly = "coo_column_seeded_callback"}

  enzymexla.jacobian_materialization materializer = @dfdyp_materializer
    residual = @component_residual
    method = <one_hot_forward>
    storage = <sparse_callback>
    fwddiff_calls = 4
    todense_calls = 8
    sparse_store_callbacks = 4
    {active_input_dimension_arg = 2 : i64,
     active_input_index = 2 : i64,
     active_output_index = 0 : i64,
     input_activity = ["enzyme_const", "enzyme_const", "enzyme_dup"],
     input_count = 3 : i64,
     output_activity = ["enzyme_dupnoneed"],
     output_count = 1 : i64,
     output_dimension_arg = 1 : i64,
     seed_loop_dimension_arg = 2 : i64,
     source = "DfDyp",
     sparse_assembly = "coo_column_seeded_callback"}

  enzymexla.sundials.ida_solve residual = @ida_residual
    jacobian = @ida_jac_callback
    linear_solver = <explicit_sparse_direct>
    jacobian_demand = <explicit_matrix>
    () {source = "llvm_sundials_ida",
        source_function = "host_configure",
        jacobian_registration_source_function = "host_linear_solver"} : () -> ()
}

// NO-BRIDGE: module attributes
// NO-BRIDGE-SAME: enzymexla.jacobian_actions_synthesized = 2 : i64
// NO-BRIDGE-NOT: enzymexla.sundials.ida_unique_host_jacobian_bridges
// NO-BRIDGE-NOT: enzymexla.sundials.ida_matrix_free_selected

// NO-BRIDGE: enzymexla.sundials.ida_solve residual = @ida_residual
// NO-BRIDGE-SAME: jacobian = @ida_jac_callback
// NO-BRIDGE-SAME: linear_solver = <explicit_sparse_direct>
// NO-BRIDGE-SAME: jacobian_demand = <explicit_matrix>
// NO-BRIDGE-SAME: source_function = "host_configure"
// NO-BRIDGE-NOT: enzymexla.sundials.unique_host_jacobian_bridge

// BRIDGE: module attributes
// BRIDGE-SAME: enzymexla.jacobian_actions_synthesized = 3 : i64
// BRIDGE-SAME: enzymexla.sundials.ida_effective_jacobian_actions_synthesized = 1 : i64
// BRIDGE-SAME: enzymexla.sundials.ida_matrix_free_selected = 1 : i64
// BRIDGE-SAME: enzymexla.sundials.ida_unique_host_jacobian_bridges = 1 : i64

// BRIDGE: enzymexla.jacobian_action @__enzymexla_sundials_ida_effective_jacobian_action_0 materialization = @dfdy_materializer residual = @component_residual active_input_index = 1 active_output_index = 0
// BRIDGE-SAME: enzymexla.sundials.ida_effective_jacobian_action
// BRIDGE-SAME: y_materialization = @dfdy_materializer
// BRIDGE-SAME: yp_materialization = @dfdyp_materializer

// BRIDGE: enzymexla.sundials.ida_solve residual = @component_residual
// BRIDGE-SAME: jacobian = @dfdy_materializer
// BRIDGE-SAME: jacobian_action = @__enzymexla_sundials_ida_effective_jacobian_action_0
// BRIDGE-SAME: linear_solver = <jacobian_action_iterative>
// BRIDGE-SAME: jacobian_demand = <jacobian_action>
// BRIDGE-SAME: bridge_host_jacobian_callback = "ida_jac_callback"
// BRIDGE-SAME: bridge_host_residual_callback = "ida_residual"
// BRIDGE-SAME: enzymexla.sundials.allow_matrix_free
// BRIDGE-SAME: enzymexla.sundials.matrix_free_selected
// BRIDGE-SAME: enzymexla.sundials.unique_host_jacobian_bridge
