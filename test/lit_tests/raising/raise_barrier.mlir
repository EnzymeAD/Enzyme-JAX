// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A block-wide barrier over a batched thread axis is a no-op in raised form:
// whole-tensor updates are already ordered, so the write to scratch is
// complete before any lane reads it back.
func.func @barrier(%out: memref<32xf64, 1>, %in: memref<32xf64, 1>) {
  %scr = memref.alloca() : memref<32xf64>
  affine.parallel (%t) = (0) to (32) {
    %v = affine.load %in[%t] : memref<32xf64, 1>
    affine.store %v, %scr[%t] : memref<32xf64>
    "enzymexla.barrier"(%t) : (index) -> ()
    %x = affine.load %scr[31 - %t] : memref<32xf64>
    affine.store %x, %out[%t] : memref<32xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @barrier_raised(
// CHECK-NOT: enzymexla.barrier
// CHECK: stablehlo.reverse
