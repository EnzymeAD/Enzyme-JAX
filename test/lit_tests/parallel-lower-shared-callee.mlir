// RUN: enzymexlamlir-opt %s --parallel-lower | FileCheck %s

// One device function launched from two sites, each by-value wrapped and
// inlined on its own.
module {
  llvm.func private @foo(%p: !llvm.ptr, %s: !llvm.ptr {llvm.byval = !llvm.struct<(i32)>}) {
    %v = llvm.load %s : !llvm.ptr -> !llvm.struct<(i32)>
    %e = llvm.extractvalue %v[0] : !llvm.struct<(i32)>
    nvvm.barrier
    llvm.store %e, %p : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @launch(%p: !llvm.ptr, %s: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %sh = arith.constant 0 : i32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) dynamic_shared_memory_size %sh {
      llvm.call @foo(%p, %s) : (!llvm.ptr, !llvm.ptr) -> ()
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) dynamic_shared_memory_size %sh {
      llvm.call @foo(%p, %s) : (!llvm.ptr, !llvm.ptr) -> ()
      gpu.terminator
    }
    llvm.return
  }
}

// CHECK-NOT: llvm.func private @foo
// CHECK-LABEL: llvm.func @launch
// CHECK: scf.parallel
// CHECK: llvm.store
// CHECK: scf.parallel
// CHECK: llvm.store
// CHECK-NOT: llvm.call
// CHECK-NOT: tmp_for_inline
