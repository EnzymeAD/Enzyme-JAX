// RUN: enzymexlamlir-opt %s -remove-duplicate-func-def | FileCheck %s

// Only duplicates whose uses are all calls are erased: other symbol uses may
// observe the identity of the function. Functions in different symbol tables
// are never merged.

module {
  // Only call uses: @f_dup is erased.
  // CHECK: llvm.func @f(
  llvm.func @f(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK-NOT: llvm.func @f_dup(
  llvm.func @f_dup(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }

  // @addr_taken has its address taken, so it is kept and @addr_only_called is
  // erased in its favor.
  // CHECK-NOT: llvm.func @addr_only_called(
  llvm.func @addr_only_called(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
  // CHECK: llvm.func @addr_taken(
  llvm.func @addr_taken(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    llvm.return %0 : i64
  }

  // Both have their address taken: both are kept.
  // CHECK: llvm.func @addr_a(
  llvm.func @addr_a(%arg0: i64) -> i64 {
    %0 = llvm.sub %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
  // CHECK: llvm.func @addr_b(
  llvm.func @addr_b(%arg0: i64) -> i64 {
    %0 = llvm.sub %arg0, %arg0 : i64
    llvm.return %0 : i64
  }

  // Both are referenced from an attribute: both are kept.
  // CHECK: llvm.func @attr_a(
  llvm.func @attr_a(%arg0: i64) -> i64 {
    %0 = llvm.xor %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
  // CHECK: llvm.func @attr_b(
  llvm.func @attr_b(%arg0: i64) -> i64 {
    %0 = llvm.xor %arg0, %arg0 : i64
    llvm.return %0 : i64
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
    // Same name as a removed top-level duplicate; calls to it must not be
    // redirected.
    // CHECK: llvm.func @f_dup(
    llvm.func @f_dup(%arg0: i64) -> i64 {
      %0 = llvm.and %arg0, %arg0 : i64
      llvm.return %0 : i64
    }
    // Same body as @f, but in a different symbol table: must be kept.
    // CHECK: llvm.func @f_inner(
    llvm.func @f_inner(%arg0: i64) -> i64 {
      llvm.return %arg0 : i64
    }

    // CHECK-LABEL: llvm.func @inner_user(
    llvm.func @inner_user(%arg0: i64) -> i64 {
      // CHECK: llvm.call @g(
      %0 = llvm.call @g_dup(%arg0) : (i64) -> i64
      // CHECK: llvm.call @f_dup(
      %1 = llvm.call @f_dup(%0) : (i64) -> i64
      llvm.return %1 : i64
    }
  }

  // CHECK-LABEL: llvm.func @user(
  // CHECK-SAME: test.ref_a = @attr_a, test.ref_b = @attr_b
  llvm.func @user(%arg0: i64) -> !llvm.ptr attributes {test.ref_a = @attr_a, test.ref_b = @attr_b} {
    // CHECK: llvm.call @f(
    %0 = llvm.call @f_dup(%arg0) : (i64) -> i64
    // CHECK: llvm.call @addr_taken(
    %1 = llvm.call @addr_only_called(%0) : (i64) -> i64
    // CHECK: llvm.call @attr_b(
    %2 = llvm.call @attr_b(%1) : (i64) -> i64
    // CHECK: llvm.mlir.addressof @addr_taken :
    %3 = llvm.mlir.addressof @addr_taken : !llvm.ptr
    // CHECK: llvm.mlir.addressof @addr_a :
    %4 = llvm.mlir.addressof @addr_a : !llvm.ptr
    // CHECK: llvm.mlir.addressof @addr_b :
    %5 = llvm.mlir.addressof @addr_b : !llvm.ptr
    llvm.return %5 : !llvm.ptr
  }
}
