// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s
// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1,cpuify{method=distribute},parallel-serialization,canonicalize)" | FileCheck %s -check-prefix=SERIAL

// A parallel op left inside the launch after mapping runs within one thread.
// A barrier inside it synchronizes that parallel's own iterations, not the
// launch's threads: it must stay an enzymexla.barrier so that cpuify can
// distribute the loop at it before it is serialized. Converting it to
// gpu.barrier both loses that structure -- naive serialization then runs
// thread-zero-guarded trailing ops on the first iteration, before the other
// iterations have executed -- and emits a divergent block-wide sync.

module {
  func.func @f(%in: memref<?xf64, 1>, %out: memref<?xf64, 1>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%t) = (%c0) to (%c256) step (%c1) {
        %alloca = memref.alloca() : memref<4xf64>
        %g = arith.cmpi slt, %t, %c3 : index
        scf.if %g {
          scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
            %v = memref.load %in[%q] : memref<?xf64, 1>
            memref.store %v, %alloca[%q] : memref<4xf64>
            scf.reduce
          }
          %a = memref.load %alloca[%c0] : memref<4xf64>
          %b = memref.load %alloca[%c1] : memref<4xf64>
          %s = arith.addf %a, %b : f64
          memref.store %s, %out[%t] : memref<?xf64, 1>
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK: scf.parallel (%[[Q:.+]]) =
// CHECK: memref.store
// CHECK-NEXT: "enzymexla.barrier"(%[[Q]])
// CHECK: scf.if
// CHECK-NOT: gpu.barrier

// The distributed form runs every iteration's store, then the guarded sum.
// SERIAL-LABEL: func.func @f(
// SERIAL: scf.for %[[I:.+]] = %c0 to %c2
// SERIAL: memref.store
// SERIAL-NEXT: }
// SERIAL: scf.for
// SERIAL: scf.if
// SERIAL: %[[A:.+]] = memref.load
// SERIAL: %[[B:.+]] = memref.load
// SERIAL: arith.addf %[[A]], %[[B]]
// SERIAL: memref.store
