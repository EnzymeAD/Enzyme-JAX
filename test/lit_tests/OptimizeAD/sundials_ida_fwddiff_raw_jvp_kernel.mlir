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
// CHECK-SAME: enzymexla.sundials.ida_raw_jvp_kernels_emitted = 1 : i64
// CHECK-SAME: enzymexla.sundials.ida_runtime_glue_emitted = 1 : i64

// CHECK-LABEL: llvm.func @__enzymexla_sundials_ida_raw_jvp_kernel_0
// CHECK-SAME: enzymexla.sundials.callback_context = "context_input_accessor"
// CHECK-SAME: enzymexla.sundials.jacobian_action = @ida_effective_action
// CHECK-SAME: enzymexla.sundials.jvp_kernel_body = "enzyme_fwddiff_raw_buffer_calls"
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

// CHECK-LABEL: enzymexla.sundials.ida_solve residual = @residual
// CHECK-SAME: enzymexla.sundials.runtime_raw_jvp_kernel = @__enzymexla_sundials_ida_raw_jvp_kernel_0
