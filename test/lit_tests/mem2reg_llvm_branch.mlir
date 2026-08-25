// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A value stored on each side of an llvm-dialect branch reaches the join as
// a block argument, the same as through the cf dialect.
llvm.func @through_llvm_cond_br(%c: i1, %a: i32, %b: i32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %p = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.cond_br %c, ^bb1, ^bb2
^bb1:
  llvm.store %a, %p : i32, !llvm.ptr
  llvm.br ^bb3
^bb2:
  llvm.store %b, %p : i32, !llvm.ptr
  llvm.br ^bb3
^bb3:
  %v = llvm.load %p : !llvm.ptr -> i32
  llvm.return %v : i32
}

// CHECK-LABEL: llvm.func @through_llvm_cond_br(
// CHECK-SAME: %{{[a-z0-9]+}}: i1, %[[A:[a-z0-9]+]]: i32, %[[B:[a-z0-9]+]]: i32
// CHECK-NOT: llvm.alloca
// CHECK: ^bb1:
// CHECK: llvm.br ^bb3(%[[A]] : i32)
// CHECK: ^bb2:
// CHECK: llvm.br ^bb3(%[[B]] : i32)
// CHECK: ^bb3(%[[V:.+]]: i32):
// CHECK: llvm.return %[[V]] : i32
