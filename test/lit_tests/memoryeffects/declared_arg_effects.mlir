// RUN: enzymexlamlir-opt --mark-func-memory-effects %s | FileCheck %s

// The effects declared on an argument are kept even if its uses do not show
// them, as for a buffer written by MPI after the function only hands MPI its
// request.
module {
  llvm.func @wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}, %arg2: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> i64
    llvm.return
  }
}

// CHECK: llvm.func @wrapper(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"], llvm.nofree}, %arg2: !llvm.ptr {enzymexla.memory_effects = [], llvm.nofree})
