// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -tessera-lower-guards -split-input-file | FileCheck %s

// A literal 0 beside a pointer is the null pointer, as in C: in a guard it is
// compared with llvm.mlir.zero, and in a pattern it matches a null argument.

module {
  tessera.define private @lib.run(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false, tessera.original_name = "lib_run"}
  tessera.define private @lib.run_default(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false, tessera.original_name = "lib_run_default"}

  tessera.optimizations {
    tessera.optimization "if opts == 0, lib.run(x, opts) -> lib.run_default(x)"
  }

  // Not known at compile time: checked at run time, against null.
  // CHECK-LABEL: llvm.func @unknown
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llvm.icmp "eq" %arg1, %[[NULL]] : !llvm.ptr
  // CHECK: tessera.call @lib.run_default(%arg0)
  // CHECK: tessera.call @lib.run(%arg0, %arg1)
  llvm.func @unknown(%x: !llvm.ptr, %opts: !llvm.ptr) -> i32 {
    %r = tessera.call @lib.run(%x, %opts) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// -----

module {
  tessera.define private @lib.run(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false, tessera.original_name = "lib_run"}
  tessera.define private @lib.run_default(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false, tessera.original_name = "lib_run_default"}

  tessera.optimizations {
    tessera.optimization "lib.run(x, 0) -> lib.run_default(x)"
  }

  // A null argument matches the literal.
  // CHECK-LABEL: llvm.func @null_arg
  // CHECK: tessera.call @lib.run_default(%arg0)
  // CHECK-NOT: tessera.call @lib.run(
  // CHECK: llvm.return
  llvm.func @null_arg(%x: !llvm.ptr) -> i32 {
    %null = llvm.mlir.zero : !llvm.ptr
    %r = tessera.call @lib.run(%x, %null) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // Any other pointer does not.
  // CHECK-LABEL: llvm.func @other_arg
  // CHECK: tessera.call @lib.run(%arg0, %arg1)
  // CHECK-NOT: lib.run_default
  // CHECK: llvm.return
  llvm.func @other_arg(%x: !llvm.ptr, %opts: !llvm.ptr) -> i32 {
    %r = tessera.call @lib.run(%x, %opts) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}
