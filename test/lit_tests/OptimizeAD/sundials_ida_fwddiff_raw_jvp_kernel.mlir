// RUN: enzymexlamlir-opt --emit-sundials-ida-runtime-glue-llvm %s | FileCheck %s

module {
  llvm.mlir.global external @enzyme_const(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  llvm.mlir.global external @enzyme_dup(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  llvm.mlir.global external @enzyme_dupnoneed(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32

  llvm.func @residual(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @__enzyme_fwddiff_y(!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr)
  llvm.func @__enzyme_fwddiff_yp(!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr)

  llvm.func @dfdy_materializer(%model: !llvm.ptr, %y: !llvm.ptr, %yp: !llvm.ptr, %wb: !llvm.ptr, %seed: !llvm.ptr, %scratch: !llvm.ptr, %out: !llvm.ptr) {
    %residual = llvm.mlir.addressof @residual : !llvm.ptr
    %const_addr = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %dup_addr = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %dupnoneed_addr = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %const = llvm.load %const_addr : !llvm.ptr -> i32
    %dup = llvm.load %dup_addr : !llvm.ptr -> i32
    %dupnoneed = llvm.load %dupnoneed_addr : !llvm.ptr -> i32
    llvm.call @__enzyme_fwddiff_y(%residual, %const, %model, %dup, %y, %seed, %const, %yp, %const, %wb, %dupnoneed, %scratch, %out) : (!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @dfdyp_materializer(%model: !llvm.ptr, %y: !llvm.ptr, %yp: !llvm.ptr, %wb: !llvm.ptr, %seed: !llvm.ptr, %scratch: !llvm.ptr, %out: !llvm.ptr) {
    %residual = llvm.mlir.addressof @residual : !llvm.ptr
    %const_addr = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %dup_addr = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %dupnoneed_addr = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %const = llvm.load %const_addr : !llvm.ptr -> i32
    %dup = llvm.load %dup_addr : !llvm.ptr -> i32
    %dupnoneed = llvm.load %dupnoneed_addr : !llvm.ptr -> i32
    llvm.call @__enzyme_fwddiff_yp(%residual, %const, %model, %const, %y, %dup, %yp, %seed, %const, %wb, %dupnoneed, %scratch, %out) : (!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  enzymexla.jacobian_action @ida_effective_action
    materialization = @dfdy_materializer
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {enzymexla.sundials.ida_effective_jacobian_action,
     input_count = 4 : i64,
     output_count = 1 : i64,
     y_materialization = @dfdy_materializer,
     yp_active_input_index = 2 : i64,
     yp_materialization = @dfdyp_materializer}

  enzymexla.sundials.ida_solve residual = @residual
    jacobian = @dfdy_materializer
    jacobian_action = @ida_effective_action
    linear_solver = <jacobian_action_iterative>
    jacobian_demand = <jacobian_action>
    () {enzymexla.sundials.matrix_free_selected} : () -> ()
}

// CHECK: module attributes
// CHECK-SAME: enzymexla.sundials.ida_host_input_providers_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_host_splice_dispatchers_emitted = 2 : i64
// CHECK-SAME: enzymexla.sundials.ida_host_splices_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_raw_jvp_kernels_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_runtime_glue_emitted = 1 : i64

// CHECK-LABEL: enzymexla.sundials.ida_host_splice @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: setup = @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: teardown = @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: registration = @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: jactimes_callback = @__enzymexla_sundials_ida_jactimes_0
// CHECK-SAME: jvp_kernel = @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK-SAME: raw_jvp_kernel = @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.runtime_context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.runtime_context_input_indices = [0, 3]
// CHECK-SAME: enzymexla.sundials.runtime_non_model_context_input_indices = [3]
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_splice_plan"
// CHECK-SAME: input_provider = @__enzymexla_sundials_ida_fill_generated_jvp_inputs
// CHECK-SAME: jacobian_action = @ida_effective_action
// CHECK-SAME: residual = @residual
// CHECK-SAME: setup_dispatcher = @__enzymexla_sundials_ida_setup_generated_jactimes
// CHECK-SAME: source = "generated_ida_jvp_host_splice"
// CHECK-SAME: teardown_dispatcher = @__enzymexla_sundials_ida_teardown_generated_jactimes

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_fill_generated_jvp_inputs
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.runtime_context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.runtime_non_model_context_input_indices = [3]
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_context_input_provider"
// CHECK: %[[INPUT_COUNT:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[FAIL:.*]] = llvm.mlir.constant(-1 : i64) : i64
// CHECK: llvm.icmp "eq" %arg1
// CHECK: llvm.return %[[INPUT_COUNT]] : i64
// CHECK: llvm.return %[[FAIL]] : i64
// CHECK: llvm.store %arg0
// CHECK: %[[RESOLVED:.*]] = llvm.call @__enzymexla_sundials_ida_resolve_generated_jvp_input
// CHECK-SAME: enzymexla.sundials.input_index = 3 : i64
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_non_model_input_resolve"
// CHECK: llvm.store %[[RESOLVED]]
// CHECK: llvm.icmp "eq" %[[RESOLVED]]
// CHECK: llvm.cond_br
// CHECK: llvm.return %[[INPUT_COUNT]] : i64

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_teardown_generated_jactimes
// CHECK-SAME: enzymexla.sundials.dispatches_teardown = @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_teardown_dispatcher"
// CHECK-SAME: enzymexla.sundials.teardown_argument = "ida_mem"
// CHECK: llvm.call @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_host_teardown_dispatch"

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_setup_generated_jactimes
// CHECK-SAME: enzymexla.sundials.dispatches_setup = @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.runtime_context_input_indices = [0, 3]
// CHECK-SAME: enzymexla.sundials.runtime_non_model_context_input_indices = [3]
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_jvp_host_setup_dispatcher"
// CHECK: llvm.call @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_host_setup_dispatch"

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_teardown_jactimes_0
// CHECK-SAME: enzymexla.sundials.context_deallocator = @__enzymexla_sundials_ida_destroy_remembered_jvp_context
// CHECK-SAME: enzymexla.sundials.teardown_argument = "ida_mem"
// CHECK: llvm.call @__enzymexla_sundials_ida_destroy_remembered_linear_solver
// CHECK-SAME: enzymexla.sundials.role = "ida_iterative_linear_solver_destroy_for_ida_mem"
// CHECK: llvm.call @__enzymexla_sundials_ida_destroy_remembered_jvp_context
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_destroy_for_ida_mem"

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_setup_jactimes_0
// CHECK-SAME: enzymexla.sundials.context_owner_registry = @__enzymexla_sundials_ida_remember_jvp_context
// CHECK-SAME: enzymexla.sundials.runtime_context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.runtime_context_input_indices = [0, 3]
// CHECK-SAME: enzymexla.sundials.runtime_non_model_context_input_indices = [3]
// CHECK: llvm.call @__enzymexla_sundials_ida_remember_jvp_context
// CHECK-SAME: enzymexla.sundials.role = "ida_jvp_context_remember"

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_register_jactimes_0
// CHECK-SAME: enzymexla.sundials.runtime_context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.runtime_context_input_indices = [0, 3]
// CHECK-SAME: enzymexla.sundials.runtime_non_model_context_input_indices = [3]
// CHECK: %[[PRETYPE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[KRYLOV_DIM64:.*]] = llvm.call @N_VGetLength(%arg1)
// CHECK-SAME: enzymexla.sundials.role = "ida_iterative_linear_solver_dimension"
// CHECK: %[[ZERO64:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[GT_ZERO:.*]] = llvm.icmp "sgt" %[[KRYLOV_DIM64]], %[[ZERO64]] : i64
// CHECK: %[[INT_MAX64:.*]] = llvm.mlir.constant(2147483647 : i64) : i64
// CHECK: %[[LE_INT_MAX:.*]] = llvm.icmp "sle" %[[KRYLOV_DIM64]], %[[INT_MAX64]] : i64
// CHECK: %[[IN_INT_RANGE:.*]] = llvm.and %[[GT_ZERO]], %[[LE_INT_MAX]] : i1
// CHECK: %[[TRUNCATED_MAXL:.*]] = llvm.trunc %[[KRYLOV_DIM64]] : i64 to i32
// CHECK: %[[DEFAULT_MAXL:.*]] = llvm.mlir.zero : i32
// CHECK: %[[MAXL:.*]] = llvm.select %[[IN_INT_RANGE]], %[[TRUNCATED_MAXL]], %[[DEFAULT_MAXL]] : i1, i32
// CHECK: %[[LS:.*]] = llvm.call @SUNLinSol_SPGMR(%arg1, %[[PRETYPE]], %[[MAXL]], %arg2)
// CHECK-SAME: enzymexla.sundials.role = "ida_iterative_linear_solver"
// CHECK: llvm.call @__enzymexla_sundials_ida_remember_linear_solver
// CHECK-SAME: %[[LS]]
// CHECK-SAME: enzymexla.sundials.role = "ida_iterative_linear_solver_remember"

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.callback_context = "context_input_accessor"
// CHECK-SAME: enzymexla.sundials.context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.context_input_indices = [0, 3]
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.jvp_kernel_body = "enzyme_fwddiff_raw_buffer_calls"
// CHECK-SAME: enzymexla.sundials.non_model_context_input_indices = [3]
// CHECK-SAME: enzymexla.sundials.raw_jvp_accumulation = "context_accumulate"
// CHECK-SAME: enzymexla.sundials.runtime_role = "ida_raw_jvp_kernel"
// CHECK: %[[CTX_MODEL:.*]] = llvm.call @__enzymexla_sundials_ida_context_input
// CHECK-SAME: enzymexla.sundials.input_index = 0 : i64
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_context_input"
// CHECK: %[[CTX_WB:.*]] = llvm.call @__enzymexla_sundials_ida_context_input
// CHECK-SAME: enzymexla.sundials.input_index = 3 : i64
// CHECK: llvm.call @__enzyme_fwddiff_y
// CHECK-SAME: %[[CTX_MODEL]]
// CHECK-SAME: %arg1
// CHECK-SAME: %arg4
// CHECK-SAME: %arg2
// CHECK-SAME: %[[CTX_WB]]
// CHECK-SAME: %arg3
// CHECK-SAME: %arg5
// CHECK-SAME: enzymexla.sundials.active_input_index = 1 : i64
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_fwddiff"
// CHECK: llvm.call @__enzyme_fwddiff_yp
// CHECK-SAME: %arg1
// CHECK-SAME: %arg2
// CHECK-SAME: %arg8
// CHECK-SAME: %arg9
// CHECK-SAME: enzymexla.sundials.active_input_index = 2 : i64
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_fwddiff"
// CHECK: %[[ACCUM:.*]] = llvm.call @__enzymexla_sundials_ida_accumulate_raw_jvp
// CHECK-SAME: enzymexla.sundials.role = "ida_raw_jvp_accumulate"
// CHECK: llvm.return %[[ACCUM]] : i32
// CHECK-NOT: semantic_raw_kernel_requires_lowering

// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_context_input(!llvm.ptr, i64) -> !llvm.ptr
// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_accumulate_raw_jvp(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_destroy_remembered_jvp_context(!llvm.ptr)
// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_remember_jvp_context(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_destroy_remembered_linear_solver(!llvm.ptr)
// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_remember_linear_solver(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__enzymexla_sundials_ida_resolve_generated_jvp_input(!llvm.ptr, i64) -> !llvm.ptr

// CHECK-LABEL: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: enzymexla.sundials.runtime_context_input_count = 4 : i64
// CHECK-SAME: enzymexla.sundials.runtime_context_input_indices = [0, 3]
// CHECK-SAME: enzymexla.sundials.runtime_host_input_provider = @__enzymexla_sundials_ida_fill_generated_jvp_inputs
// CHECK-SAME: enzymexla.sundials.runtime_host_setup_dispatcher = @__enzymexla_sundials_ida_setup_generated_jactimes
// CHECK-SAME: enzymexla.sundials.runtime_host_splice = @__enzymexla_sundials_ida_host_splice_0
// CHECK-SAME: enzymexla.sundials.runtime_host_teardown_dispatcher = @__enzymexla_sundials_ida_teardown_generated_jactimes
// CHECK-SAME: enzymexla.sundials.runtime_non_model_context_input_indices = [3]
// CHECK-SAME: enzymexla.sundials.runtime_raw_jvp_kernel = @__enzymexla_sundials_ida_raw_jvp_kernel_0
