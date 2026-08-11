// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Splitting a kernel whose thread count does not divide its work leaves the
// block parallel inside the guard its bounds need. The launch wants that
// parallel directly in the grid parallel's body, so without interchanging the
// two the wrapper is never converted and survives to the LLVM lowering, where
// it fails as an unhandled unrealized conversion cast. The condition is
// computed outside the block parallel, so it is the same for every thread:
// running the parallel and having each thread skip the body is what the guard
// already means.

module {
  func.func @guarded(%n: index, %m: index, %out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c8, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%n) step (%c1) {
        %in = arith.cmpi slt, %i, %m : index
        scf.if %in {
          %off = arith.muli %i, %c8 : index
          scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
            %k = arith.addi %off, %j : index
            memref.store %v, %out[%k] : memref<?xf64>
            scf.reduce
          }
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// The guard ends up inside the thread parallel, and the kernel becomes a
// launch instead of surviving as a wrapper.
// CHECK-LABEL: func.func @guarded
// CHECK-NOT: enzymexla.gpu_wrapper
// CHECK: gpu.launch blocks
// CHECK: scf.if
