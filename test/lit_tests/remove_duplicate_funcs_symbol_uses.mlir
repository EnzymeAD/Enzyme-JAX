// RUN: enzymexlamlir-opt %s -remove-duplicate-func-def | FileCheck %s

// Every symbol reference to a removed duplicate is redirected, not only call
// sites, and functions in different symbol tables are never merged.

module {
  // CHECK: llvm.func @f(
  llvm.func @f(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }

  // CHECK-NOT: llvm.func @f_dup(
  llvm.func @f_dup(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }

  // CHECK: module @inner
  module @inner {
    // CHECK: llvm.func @g(
    llvm.func @g(%arg0: i64) -> i64 {
      %0 = llvm.add %arg0, %arg0 : i64
      llvm.return %0 : i64
    }
    // CHECK-NOT: llvm.func @g_dup(
    llvm.func @g_dup(%arg0: i64) -> i64 {
      %0 = llvm.add %arg0, %arg0 : i64
      llvm.return %0 : i64
    }
    // Same name as a removed top-level duplicate; references to it must not be
    // redirected.
    // CHECK: llvm.func @f_dup(
    llvm.func @f_dup() {
      llvm.return
    }
    // Same body as @f, but in a different symbol table: must be kept.
    // CHECK: llvm.func @f_inner(
    llvm.func @f_inner(%arg0: i64) -> i64 {
      llvm.return %arg0 : i64
    }
  }

  // CHECK-LABEL: llvm.func @user
  // CHECK-SAME: test.inner_ref = @inner::@f_dup, test.nested_ref = @inner::@g, test.ref = @f}
  llvm.func @user() -> !llvm.ptr attributes {test.ref = @f_dup, test.nested_ref = @inner::@g_dup, test.inner_ref = @inner::@f_dup} {
    // CHECK: llvm.mlir.addressof @f :
    %0 = llvm.mlir.addressof @f_dup : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
