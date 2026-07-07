// RUN: enzymexlamlir-opt --emit-sundials-ida-runtime-glue-llvm --strip-sundials-ida-runtime-glue-metadata %s | FileCheck %s

module attributes {
  enzymexla.sundials.ida_runtime_glue_emitted = 99 : i64,
  gridkit.jacobian.marked_sparse_helpers = 4 : i64
} {
  llvm.func @residual(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @materialize_sparse_jacobian(!llvm.ptr, !llvm.ptr, !llvm.ptr)

  enzymexla.jacobian_action @ida_effective_action
    materialization = @materialize_sparse_jacobian
    residual = @residual
    active_input_index = 1
    active_output_index = 0
    {enzymexla.sundials.ida_effective_jacobian_action,
     input_count = 4 : i64,
     output_count = 1 : i64,
     source = "unit-test"}

  enzymexla.sundials.ida_solve residual = @residual
    jacobian = @materialize_sparse_jacobian
    jacobian_action = @ida_effective_action
    linear_solver = <jacobian_action_iterative>
    jacobian_demand = <jacobian_action>
    () {enzymexla.sundials.matrix_free_selected,
        source = "unit-test"} : () -> ()
}

// CHECK-LABEL: module
// CHECK-NOT: enzymexla.
// CHECK-NOT: gridkit.
// CHECK: llvm.func @__enzymexla_sundials_ida_fill_generated_jvp_inputs
// CHECK: llvm.func @__enzymexla_sundials_ida_teardown_generated_jactimes
// CHECK: llvm.func @__enzymexla_sundials_ida_setup_generated_jactimes
// CHECK: llvm.func @__enzymexla_sundials_ida_register_jactimes_0
// CHECK: llvm.func @__enzymexla_sundials_ida_jactimes_0
// CHECK: llvm.func @__enzymexla_sundials_ida_jvp_kernel_0
// CHECK: llvm.func @IDASetJacTimes
// CHECK-NOT: enzymexla.
// CHECK-NOT: gridkit.
