// RUN: enzymexlamlir-opt --split-input-file --emit-sundials-ida-runtime-glue-llvm %s | FileCheck %s

module {
  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @materialize_sparse_jacobian() {
    return
  }

  enzymexla.jacobian_action @ida_effective_action
    materialization = @materialize_sparse_jacobian
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {enzymexla.sundials.ida_effective_jacobian_action,
     source = "unit-test"}

  func.func @selected_solve(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @materialize_sparse_jacobian
      jacobian_action = @ida_effective_action
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%y, %yp) {enzymexla.sundials.matrix_free_selected,
                 jacobian_registration_source_function = "host_configure_linear_solver",
                 linear_solver_source_function = "host_configure_linear_solver",
                 residual_registration_source_function = "host_configure",
                 source_function = "host_configure"}
      : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }

  func.func @explicit_solve(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian = @materialize_sparse_jacobian
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%y, %yp) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %state : tensor<4xf64>
  }
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.sundials.ida_host_splices_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_jvp_kernel_adapters_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_raw_jvp_kernels_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_runtime_glue_emitted = 1 : i64

// CHECK-LABEL: enzymexla.sundials.ida_host_splice @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: setup = @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: teardown = @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: registration = @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: jactimes_callback = @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: jvp_kernel = @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: raw_jvp_kernel = @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.host_jacobian_registration_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_linear_solver_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_residual_registration_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_splice_plan"
// CHECK-SAME: jacobian_action = @ida_effective_action
// CHECK-SAME: residual = @residual
// CHECK-SAME: source = "generated_ida_jvp_host_splice"

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: enzymexla.sundials.context_deallocator = @__enzymexla_sundials_ida_destroy_jvp_context
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.host_jacobian_registration_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_linear_solver_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_residual_registration_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_context_teardown"
// CHECK-SAME: enzymexla.sundials.source_function = "host_configure"
// CHECK: llvm.call @__enzymexla_sundials_ida_destroy_jvp_context
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_destroy"
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: enzymexla.sundials.context_allocator = @__enzymexla_sundials_ida_create_jvp_context
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.host_jacobian_registration_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_linear_solver_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_residual_registration_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.runtime_registration = @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_context_setup"
// CHECK-SAME: enzymexla.sundials.source_function = "host_configure"
// CHECK: %[[OUT_SIZE:.*]] = llvm.call @N_VGetLength
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_output_size"
// CHECK: %[[CTX:.*]] = llvm.call @__enzymexla_sundials_ida_create_jvp_context
// CHECK-SAME: %[[OUT_SIZE]]
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_create"
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK: llvm.store %[[CTX]], %arg6 : !llvm.ptr, !llvm.ptr
// CHECK: %[[SETUP_STATUS:.*]] = llvm.call @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: %[[CTX]]
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_registration_helper"
// CHECK: llvm.return %[[SETUP_STATUS]] : i32

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: enzymexla.sundials.callback_context = "ida_jvp_user_data_context"
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.host_jacobian_registration_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_linear_solver_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_residual_registration_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.jactimes_callback = @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: enzymexla.sundials.linear_solver = "SUNLinSol_SPGMR"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jactimes_registration"
// CHECK-SAME: enzymexla.sundials.source_function = "host_configure"
// CHECK: %[[PRETYPE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[MAXL:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @__enzymexla_sundials_ida_register_jvp_context
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_registration"
// CHECK: llvm.call @IDASetUserData
// CHECK-SAME: enzymexla.sundials.role = "ida_user_data_registration"
// CHECK: %[[LS:.*]] = llvm.call @SUNLinSol_SPGMR
// CHECK-SAME: enzymexla.sundials.role = "ida_iterative_linear_solver"
// CHECK: llvm.call @IDASetLinearSolver
// CHECK-SAME: enzymexla.sundials.role = "ida_linear_solver_registration"
// CHECK: llvm.mlir.addressof @__enzymexla_sundials_ida_jactimes_0 : !llvm.ptr
// CHECK: %[[JT:.*]] = llvm.call @IDASetJacTimes
// CHECK-SAME: enzymexla.sundials.role = "ida_jacobian_action_registration"
// CHECK: llvm.return %[[JT]] : i32

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: enzymexla.sundials.callback_body = "delegates_jvp_kernel"
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.host_jacobian_registration_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_linear_solver_source_function = "host_configure_linear_solver"
// CHECK-SAME: enzymexla.sundials.host_residual_registration_source_function = "host_configure"
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.jvp_kernel = @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.residual = @residual
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jactimes_callback"
// CHECK-SAME: enzymexla.sundials.source_function = "host_configure"
// CHECK: %[[ADAPTER_STATUS:.*]] = llvm.call @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.role = "ida_jacobian_action_jvp_kernel"
// CHECK: llvm.return %[[ADAPTER_STATUS]] : i32

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.active_output_index = 0 : i64
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.jvp_kernel_body = "nvector_unpack_and_raw_jvp_call"
// CHECK-SAME: enzymexla.sundials.materialization = @materialize_sparse_jacobian
// CHECK-SAME: enzymexla.sundials.raw_jvp_kernel = @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.residual = @residual
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_kernel_adapter"
// CHECK-SAME: enzymexla.sundials.y_active_input_index = 1 : i64
// CHECK-SAME: enzymexla.sundials.yp_tangent = "tmp1 = cj * v"
// CHECK: llvm.call @N_VScale
// CHECK-SAME: enzymexla.sundials.role = "ida_yp_tangent_scale"
// CHECK: %[[Y:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[YP:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[RR:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[V:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[JV:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[TMP1:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[TMP2:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[RAW_STATUS:.*]] = llvm.call @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_kernel"
// CHECK-SAME: (f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.return %[[RAW_STATUS]] : i32

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.jvp_kernel_body = "semantic_raw_kernel_requires_lowering"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_raw_jvp_kernel"
// CHECK: %[[UNIMPLEMENTED:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.return %[[UNIMPLEMENTED]] : i32

// CHECK: llvm.func @__enzymexla_sundials_ida_destroy_jvp_context(!llvm.ptr)
// CHECK: llvm.func @__enzymexla_sundials_ida_create_jvp_context(!llvm.ptr, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK: llvm.func @__enzymexla_sundials_ida_register_jvp_context(!llvm.ptr)
// CHECK: llvm.func @N_VScale(f64, !llvm.ptr, !llvm.ptr)
// CHECK: llvm.func @N_VGetLength(!llvm.ptr) -> i64
// CHECK: llvm.func @N_VGetArrayPointer(!llvm.ptr) -> !llvm.ptr
// CHECK: llvm.func @IDASetJacTimes(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.func @IDASetUserData(!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.func @IDASetLinearSolver(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.func @SUNLinSol_SPGMR(!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr

// CHECK-LABEL: func.func @selected_solve
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian = @materialize_sparse_jacobian
// CHECK-SAME: jacobian_action = @ida_effective_action
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: enzymexla.sundials.matrix_free_selected
// CHECK-SAME: enzymexla.sundials.runtime_context_setup = @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_context_teardown = @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_host_splice = @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: enzymexla.sundials.runtime_jactimes_callback = @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_jvp_kernel = @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.runtime_raw_jvp_kernel = @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.runtime_registration = @__enzymexla_sundials_ida_register_jactimes_0

// CHECK-LABEL: func.func @explicit_solve
// CHECK: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: linear_solver = <explicit_sparse_direct>
// CHECK-SAME: jacobian_demand = <explicit_matrix>

// -----

module {
  llvm.func @residual_callback(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  llvm.func @lowered_ida_jvp_kernel(f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {enzymexla.sundials.runtime_role = "lowered_jvp_kernel"}

  enzymexla.sundials.ida_solve residual = @residual_callback
    jacobian_action = @lowered_ida_jvp_kernel
    linear_solver = <jacobian_action_iterative>
    jacobian_demand = <jacobian_action>
    () {enzymexla.sundials.matrix_free_selected,
        source_function = "host_with_lowered_jvp_kernel"} : () -> ()
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.sundials.ida_host_splices_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_runtime_glue_emitted = 1 : i64

// CHECK-LABEL: enzymexla.sundials.ida_host_splice @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: setup = @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: registration = @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: jactimes_callback = @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_with_lowered_jvp_kernel"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_splice_plan"
// CHECK-SAME: jacobian_action = @lowered_ida_jvp_kernel
// CHECK-SAME: residual = @residual_callback

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: enzymexla.sundials.callback_body = "delegates_jvp_kernel"
// CHECK-SAME: enzymexla.sundials.jacobian_action = @lowered_ida_jvp_kernel
// CHECK-SAME: enzymexla.sundials.jvp_kernel = @lowered_ida_jvp_kernel
// CHECK-SAME: enzymexla.sundials.residual = @residual_callback
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jactimes_callback"
// CHECK-SAME: enzymexla.sundials.source_function = "host_with_lowered_jvp_kernel"
// CHECK: %[[STATUS:.*]] = llvm.call @lowered_ida_jvp_kernel
// CHECK-SAME: enzymexla.sundials.role = "ida_jacobian_action_jvp_kernel"
// CHECK-SAME: (f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.return %[[STATUS]] : i32

// CHECK-LABEL: enzymexla.sundials.ida_solve residual = @residual_callback
// CHECK-SAME: jacobian_action = @lowered_ida_jvp_kernel
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: enzymexla.sundials.runtime_host_splice = @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: enzymexla.sundials.runtime_jactimes_callback = @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_registration = @__enzymexla_sundials_ida_register_jactimes_0

// -----

module {
  llvm.func @semantic_lowered_raw_jvp_kernel(f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {enzymexla.sundials.runtime_role = "lowered_raw_jvp_kernel"}

  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @materialize_sparse_jacobian() {
    return
  }

  enzymexla.jacobian_action @semantic_action_with_raw_kernel
    materialization = @materialize_sparse_jacobian
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {enzymexla.sundials.lowered_raw_jvp_kernel = @semantic_lowered_raw_jvp_kernel,
     source = "unit-test-lowered-raw"}

  enzymexla.sundials.ida_solve residual = @residual
    jacobian = @materialize_sparse_jacobian
    jacobian_action = @semantic_action_with_raw_kernel
    linear_solver = <jacobian_action_iterative>
    jacobian_demand = <jacobian_action>
    () {enzymexla.sundials.matrix_free_selected,
        source_function = "host_with_lowered_raw_jvp_kernel"} : () -> ()
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.sundials.ida_host_splices_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_jvp_kernel_adapters_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_lowered_raw_jvp_kernels_linked = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_runtime_glue_emitted = 1 : i64

// CHECK-LABEL: enzymexla.sundials.ida_host_splice @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: raw_jvp_kernel = @semantic_lowered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_with_lowered_raw_jvp_kernel"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_splice_plan"
// CHECK-SAME: jacobian_action = @semantic_action_with_raw_kernel
// CHECK-SAME: residual = @residual

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.jacobian_action = @semantic_action_with_raw_kernel
// CHECK-SAME: enzymexla.sundials.jvp_kernel_body = "nvector_unpack_and_raw_jvp_call"
// CHECK-SAME: enzymexla.sundials.raw_jvp_kernel = @semantic_lowered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_kernel_adapter"
// CHECK-SAME: enzymexla.sundials.yp_tangent = "tmp1 = cj * v"
// CHECK: llvm.call @N_VScale
// CHECK-SAME: enzymexla.sundials.role = "ida_yp_tangent_scale"
// CHECK: %[[Y:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[YP:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[RR:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[V:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[JV:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[TMP1:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[TMP2:.*]] = llvm.call @N_VGetArrayPointer
// CHECK: %[[RAW_STATUS:.*]] = llvm.call @semantic_lowered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_kernel"
// CHECK: llvm.return %[[RAW_STATUS]] : i32
// CHECK-NOT: llvm.func @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK: llvm.func @N_VGetArrayPointer(!llvm.ptr) -> !llvm.ptr

// CHECK-LABEL: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian_action = @semantic_action_with_raw_kernel
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: enzymexla.sundials.runtime_host_splice = @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: enzymexla.sundials.runtime_jvp_kernel = @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.runtime_raw_jvp_kernel = @semantic_lowered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.runtime_registration = @__enzymexla_sundials_ida_register_jactimes_0

// -----

module {
  llvm.func @discovered_raw_jvp_kernel(f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {enzymexla.sundials.jacobian_action = @semantic_action_discovered_raw, enzymexla.sundials.runtime_role = "lowered_raw_jvp_kernel"}

  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @materialize_sparse_jacobian() {
    return
  }

  enzymexla.jacobian_action @semantic_action_discovered_raw
    materialization = @materialize_sparse_jacobian
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {source = "unit-test-discovered-raw"}

  enzymexla.sundials.ida_solve residual = @residual
    jacobian = @materialize_sparse_jacobian
    jacobian_action = @semantic_action_discovered_raw
    linear_solver = <jacobian_action_iterative>
    jacobian_demand = <jacobian_action>
    () {enzymexla.sundials.matrix_free_selected,
        source_function = "host_with_discovered_raw_jvp_kernel"} : () -> ()
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.sundials.ida_host_splices_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_jvp_kernel_adapters_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_lowered_raw_jvp_kernels_linked = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_runtime_glue_emitted = 1 : i64

// CHECK-LABEL: enzymexla.sundials.ida_host_splice @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: raw_jvp_kernel = @discovered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.host_configure_source_function = "host_with_discovered_raw_jvp_kernel"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_splice_plan"
// CHECK-SAME: jacobian_action = @semantic_action_discovered_raw
// CHECK-SAME: residual = @residual

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.jacobian_action = @semantic_action_discovered_raw
// CHECK-SAME: enzymexla.sundials.jvp_kernel_body = "nvector_unpack_and_raw_jvp_call"
// CHECK-SAME: enzymexla.sundials.raw_jvp_kernel = @discovered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_kernel_adapter"
// CHECK: %[[RAW_STATUS:.*]] = llvm.call @discovered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_kernel"
// CHECK: llvm.return %[[RAW_STATUS]] : i32
// CHECK-NOT: llvm.func @__enzymexla_sundials_ida_raw_jvp_kernel_0

// CHECK-LABEL: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: jacobian_action = @semantic_action_discovered_raw
// CHECK-SAME: linear_solver = <jacobian_action_iterative>
// CHECK-SAME: jacobian_demand = <jacobian_action>
// CHECK-SAME: enzymexla.sundials.runtime_host_splice = @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: enzymexla.sundials.runtime_jvp_kernel = @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.runtime_raw_jvp_kernel = @discovered_raw_jvp_kernel
// CHECK-SAME: enzymexla.sundials.runtime_registration = @__enzymexla_sundials_ida_register_jactimes_0
