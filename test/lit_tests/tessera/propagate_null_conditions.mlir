// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// A null pointer is 0 in a condition, as in C. @make_integ(this, ir) gives its
// object "affine" only when ir is null: built with no rule of its own.

module {
  tessera.define private @lib.use(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false}
  llvm.func @make_integ(!llvm.ptr, !llvm.ptr) attributes {tessera.conditional_establishes = [{arg = 0 : i64, condition = "ir == 0", constants = {}, names = {ir = 1 : i64}, property = "affine"}]}
  llvm.func @make_integ_ne(!llvm.ptr, !llvm.ptr) attributes {tessera.conditional_establishes = [{arg = 0 : i64, condition = "ir != 0", constants = {}, names = {ir = 1 : i64}, property = "affine"}]}
  llvm.func @alloc() -> !llvm.ptr

  // A literal null: proven.
  // CHECK-LABEL: llvm.func @null_rule
  // CHECK: tessera.call @lib.use({{.*}}) <arg_attrs = [{tessera.property = ["affine"]}]>
  llvm.func @null_rule() -> i32 {
    %null = llvm.mlir.zero : !llvm.ptr
    %p = llvm.call @alloc() : () -> !llvm.ptr
    llvm.call @make_integ(%p, %null) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @lib.use(%p) : (!llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // A rule that may be anything: not proven.
  // CHECK-LABEL: llvm.func @some_rule
  // CHECK: tessera.call @lib.use
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @some_rule(%ir: !llvm.ptr) -> i32 {
    %p = llvm.call @alloc() : () -> !llvm.ptr
    llvm.call @make_integ(%p, %ir) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @lib.use(%p) : (!llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // "ir != 0" of a null pointer is false.
  // CHECK-LABEL: llvm.func @null_ne
  // CHECK: tessera.call @lib.use
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @null_ne() -> i32 {
    %null = llvm.mlir.zero : !llvm.ptr
    %p = llvm.call @alloc() : () -> !llvm.ptr
    llvm.call @make_integ_ne(%p, %null) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @lib.use(%p) : (!llvm.ptr) -> i32
    llvm.return %r : i32
  }
}
