// RUN: enzymexlamlir-opt %s --parallel-lower --verify-diagnostics

// Pre-inlining descends into the callee's own calls before inlining it, so a
// recursive callee would recurse forever. A callee already being inlined
// keeps its recursive call sites as calls, and the pass then reports the
// call it could not remove instead of never finishing.
module {
  llvm.func private @rec(%p: !llvm.ptr, %n: i32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    nvvm.barrier
    %cmp = llvm.icmp "sgt" %n, %c0 : i32
    llvm.cond_br %cmp, ^bb1, ^bb2
  ^bb1:
    %n1 = llvm.sub %n, %c1 : i32
    llvm.store %n, %p : i32, !llvm.ptr
    // expected-error @below {{Could not erase function with gpu-specific instruction due to this use}}
    llvm.call @rec(%p, %n1) : (!llvm.ptr, i32) -> ()
    llvm.br ^bb2
  ^bb2:
    llvm.return
  }
  llvm.func @launch(%p: !llvm.ptr, %n: i32) {
    %c1 = arith.constant 1 : index
    %sh = arith.constant 0 : i32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) dynamic_shared_memory_size %sh {
      llvm.call @rec(%p, %n) : (!llvm.ptr, i32) -> ()
      gpu.terminator
    }
    llvm.return
  }
}
