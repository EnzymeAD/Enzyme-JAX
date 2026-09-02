// RUN: enzymexlamlir-opt %s --parallel-lower | FileCheck %s

// A callee already prepared for GPU inlining (a by-value wrapper this pass
// itself builds, left behind by an earlier run) needs no second wrapper: the
// call is inlined whole, and nothing may touch it afterwards.
module {
  llvm.func private @foo(%p: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    nvvm.barrier
    llvm.store %c1, %p : i32, !llvm.ptr
    llvm.return
  }
  llvm.func private @"foo$tmp_for_inline"(%p: !llvm.ptr) attributes {enzyme.prepared_for_inline} {
    llvm.call @foo(%p) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @launch(%p: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %sh = arith.constant 0 : i32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) dynamic_shared_memory_size %sh {
      llvm.call @"foo$tmp_for_inline"(%p) : (!llvm.ptr) -> ()
      gpu.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @launch
// CHECK: scf.parallel
// CHECK: scf.parallel
// CHECK: llvm.store
// CHECK-NOT: llvm.call
