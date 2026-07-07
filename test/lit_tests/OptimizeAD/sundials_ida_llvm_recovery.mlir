// RUN: enzymexlamlir-opt --recover-sundials-ida-llvm %s | FileCheck %s

module {
  llvm.func @IDAInit(!llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @IDASetUserData(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @IDASetLinearSolver(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @IDASetJacFn(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @IDASetJacTimes(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @IDASetPreconditioner(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @SUNLinSol_KLU(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @SUNLinSol_Dense(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @SUNLinSol_SPGMR(!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr

  llvm.func @residual(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @jacobian(!llvm.ptr, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @jactimes(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @preconditioner(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: module attributes {enzymexla.sundials.ida_solves_recovered = 4 : i64}

  // CHECK: enzymexla.sundials.ida_solve residual = @residual
  // CHECK-SAME: jacobian = @jacobian
  // CHECK-SAME: linear_solver = <explicit_sparse_direct>
  // CHECK-SAME: jacobian_demand = <explicit_matrix>
  // CHECK-SAME: enzymexla.sundials.user_data_registered
  // CHECK-SAME: source = "llvm_sundials_ida"
  // CHECK-SAME: source_function = "host_sparse"

  // CHECK: enzymexla.sundials.ida_solve residual = @residual
  // CHECK-SAME: jacobian_action = @jactimes
  // CHECK-SAME: preconditioner = @preconditioner
  // CHECK-SAME: linear_solver = <jacobian_action_iterative>
  // CHECK-SAME: jacobian_demand = <jacobian_action>
  // CHECK-SAME: enzymexla.sundials.user_data_registered
  // CHECK-SAME: source = "llvm_sundials_ida"
  // CHECK-SAME: source_function = "host_iterative"

  // CHECK: enzymexla.sundials.ida_solve residual = @residual
  // CHECK-SAME: linear_solver = <explicit_dense_direct>
  // CHECK-SAME: jacobian_demand = <none>
  // CHECK-SAME: enzymexla.sundials.user_data_registered
  // CHECK-SAME: source = "llvm_sundials_ida"
  // CHECK-SAME: source_function = "host_dense"

  // CHECK: enzymexla.sundials.ida_solve residual = @residual
  // CHECK-SAME: jacobian = @jacobian
  // CHECK-SAME: linear_solver = <explicit_sparse_direct>
  // CHECK-SAME: jacobian_demand = <explicit_matrix>
  // CHECK-SAME: enzymexla.sundials.user_data_registered
  // CHECK-SAME: source = "llvm_sundials_ida"
  // CHECK-SAME: source_function = "host_split"

  // CHECK-LABEL: llvm.func @host_sparse
  llvm.func @host_sparse(%mem: !llvm.ptr, %yy: !llvm.ptr, %yp: !llvm.ptr, %mat: !llvm.ptr, %ctx: !llvm.ptr) {
    %residual = llvm.mlir.addressof @residual : !llvm.ptr
    %jacobian = llvm.mlir.addressof @jacobian : !llvm.ptr
    %t0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    // CHECK: llvm.call @IDAInit
    // CHECK-SAME: enzymexla.sundials.role = "ida_residual_registration"
    %0 = llvm.call @IDAInit(%mem, %residual, %t0, %yy, %yp) : (!llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetUserData
    // CHECK-SAME: enzymexla.sundials.role = "ida_user_data_registration"
    %user = llvm.call @IDASetUserData(%mem, %ctx) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @SUNLinSol_KLU
    // CHECK-SAME: enzymexla.sundials.role = "ida_sparse_direct_linear_solver"
    %ls = llvm.call @SUNLinSol_KLU(%yy, %mat, %ctx) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    // CHECK: llvm.call @IDASetLinearSolver
    // CHECK-SAME: enzymexla.sundials.role = "ida_linear_solver_registration"
    %1 = llvm.call @IDASetLinearSolver(%mem, %ls, %mat) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetJacFn
    // CHECK-SAME: enzymexla.sundials.role = "ida_jacobian_registration"
    %2 = llvm.call @IDASetJacFn(%mem, %jacobian) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }

  // CHECK-LABEL: llvm.func @host_iterative
  llvm.func @host_iterative(%mem: !llvm.ptr, %yy: !llvm.ptr, %yp: !llvm.ptr, %ctx: !llvm.ptr) {
    %residual = llvm.mlir.addressof @residual : !llvm.ptr
    %jactimes = llvm.mlir.addressof @jactimes : !llvm.ptr
    %preconditioner = llvm.mlir.addressof @preconditioner : !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %t0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %solver_kind = llvm.mlir.constant(0 : i32) : i32
    %maxl = llvm.mlir.constant(20 : i32) : i32
    %0 = llvm.call @IDAInit(%mem, %residual, %t0, %yy, %yp) : (!llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetUserData
    // CHECK-SAME: enzymexla.sundials.role = "ida_user_data_registration"
    %user = llvm.call @IDASetUserData(%mem, %ctx) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @SUNLinSol_SPGMR
    // CHECK-SAME: enzymexla.sundials.role = "ida_iterative_linear_solver"
    %ls = llvm.call @SUNLinSol_SPGMR(%yy, %solver_kind, %maxl, %ctx) : (!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
    %1 = llvm.call @IDASetLinearSolver(%mem, %ls, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetJacTimes
    // CHECK-SAME: enzymexla.sundials.role = "ida_jacobian_action_registration"
    %2 = llvm.call @IDASetJacTimes(%mem, %null, %jactimes) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetPreconditioner
    // CHECK-SAME: enzymexla.sundials.role = "ida_preconditioner_registration"
    %3 = llvm.call @IDASetPreconditioner(%mem, %null, %preconditioner) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }

  // CHECK-LABEL: llvm.func @host_dense
  llvm.func @host_dense(%mem: !llvm.ptr, %yy: !llvm.ptr, %yp: !llvm.ptr, %mat: !llvm.ptr, %ctx: !llvm.ptr) {
    %residual = llvm.mlir.addressof @residual : !llvm.ptr
    %t0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %0 = llvm.call @IDAInit(%mem, %residual, %t0, %yy, %yp) : (!llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetUserData
    // CHECK-SAME: enzymexla.sundials.role = "ida_user_data_registration"
    %user = llvm.call @IDASetUserData(%mem, %ctx) : (!llvm.ptr, !llvm.ptr) -> i32
    %ls = llvm.call @SUNLinSol_Dense(%yy, %mat, %ctx) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %1 = llvm.call @IDASetLinearSolver(%mem, %ls, %mat) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }

  // CHECK-LABEL: llvm.func @host_split
  llvm.func @host_split(%mem: !llvm.ptr, %yy: !llvm.ptr, %yp: !llvm.ptr, %mat: !llvm.ptr, %ctx: !llvm.ptr) {
    %residual = llvm.mlir.addressof @residual : !llvm.ptr
    %t0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %0 = llvm.call @IDAInit(%mem, %residual, %t0, %yy, %yp) : (!llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.call @IDASetUserData
    // CHECK-SAME: enzymexla.sundials.role = "ida_user_data_registration"
    %user = llvm.call @IDASetUserData(%mem, %ctx) : (!llvm.ptr, !llvm.ptr) -> i32
    %1 = llvm.call @configure_sparse(%mem, %yy, %mat, %ctx) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }

  // CHECK-LABEL: llvm.func @configure_sparse
  llvm.func @configure_sparse(%mem: !llvm.ptr, %yy: !llvm.ptr, %mat: !llvm.ptr, %ctx: !llvm.ptr) -> i32 {
    %jacobian = llvm.mlir.addressof @jacobian : !llvm.ptr
    %ls = llvm.call @SUNLinSol_KLU(%yy, %mat, %ctx) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %0 = llvm.call @IDASetLinearSolver(%mem, %ls, %mat) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %1 = llvm.call @IDASetJacFn(%mem, %jacobian) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %1 : i32
  }
}
