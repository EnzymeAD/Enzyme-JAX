// RUN: not enzymexlamlir-opt --emit-sundials-ida-runtime-glue-llvm %s 2>&1 | FileCheck %s

module {
  llvm.func @raw0(f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {enzymexla.sundials.jacobian_action = @semantic_action, enzymexla.sundials.runtime_role = "lowered_raw_jvp_kernel"}
  llvm.func @raw1(f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {enzymexla.sundials.jacobian_action = @semantic_action, enzymexla.sundials.runtime_role = "lowered_raw_jvp_kernel"}

  func.func @residual(%y: tensor<4xf64>, %yp: tensor<4xf64>) -> tensor<4xf64> {
    return %y : tensor<4xf64>
  }

  func.func @materialize_sparse_jacobian() {
    return
  }

  enzymexla.jacobian_action @semantic_action
    materialization = @materialize_sparse_jacobian
    residual = @residual
    active_input_index = 1
    active_output_index = 0

  enzymexla.sundials.ida_solve residual = @residual
    jacobian = @materialize_sparse_jacobian
    jacobian_action = @semantic_action
    linear_solver = <jacobian_action_iterative>
    jacobian_demand = <jacobian_action>
    () {enzymexla.sundials.matrix_free_selected} : () -> ()
}

// CHECK: error: multiple lowered raw IDA JVP kernels advertise jacobian action @semantic_action: @raw0 @raw1
