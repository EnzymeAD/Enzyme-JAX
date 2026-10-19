// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Ops that follow an inner parallel read what every lane wrote, so when they
// are moved into the parallel under a thread-zero if, a barrier must precede
// the if: without one thread zero races ahead of the other lanes' stores, and
// when the parallel is later serialized the guarded ops run on the first
// iteration, before the remaining iterations have stored their values.

module {
  func.func @f(%in: memref<?xf64, 1>, %out: memref<?xf64, 1>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%e) = (%c0) to (%c3) step (%c1) {
        %alloca = memref.alloca() : memref<4xf64>
        scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
          %v = memref.load %in[%q] : memref<?xf64, 1>
          memref.store %v, %alloca[%q] : memref<4xf64>
          scf.reduce
        }
        %a = memref.load %alloca[%c0] : memref<4xf64>
        %b = memref.load %alloca[%c1] : memref<4xf64>
        %s = arith.addf %a, %b : f64
        memref.store %s, %out[%e] : memref<?xf64, 1>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK: %[[TID:.+]] = gpu.thread_id x
// CHECK: memref.store
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[COND:.+]] = arith.cmpi eq, %[[TID]]
// CHECK-NEXT: scf.if %[[COND]]
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.store
