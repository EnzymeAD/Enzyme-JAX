// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A thread axis whose extent is dynamic but clamped by a min against a
// constant (MFEM block sizes arrive as min(1 << log2(N), 256)) batches at
// the bound behind an `iv < extent` guard instead of peeling to a serial
// loop; the barrier over it stays a batched no-op.
func.func @bounded(%out: memref<256xf64, 1>, %in: memref<256xf64, 1>, %nbuf: memref<i32, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %c256_i32 = arith.constant 256 : i32
  %n = affine.load %nbuf[] : memref<i32, 1>
  %b = arith.minsi %n, %c256_i32 : i32
  %bi = arith.index_cast %b : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %bi, %c1, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      %scr = memref.alloca() : memref<256xf64>
      affine.parallel (%t) = (0) to (symbol(%bi)) {
        %v = affine.load %in[%t] : memref<256xf64, 1>
        affine.store %v, %scr[%t] : memref<256xf64>
        "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
        %w = affine.load %scr[0] : memref<256xf64>
        affine.store %w, %out[%t] : memref<256xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func private @rxla$raised_0(
// CHECK-NOT: stablehlo.while
// CHECK: stablehlo.select
