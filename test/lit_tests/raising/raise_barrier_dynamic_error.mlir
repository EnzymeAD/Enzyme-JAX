// RUN: not enzymexlamlir-opt %s --raise-affine-to-stablehlo 2>&1 | FileCheck %s

// A barrier over a statically sized thread axis batches away; over a
// dynamically sized axis the loop raises serialized, where dropping the
// barrier would let one lane run ahead of the others' stores: report it
// instead of miscompiling.
func.func @dynbarrier(%out: memref<?xf64, 1>, %in: memref<?xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  %c0 = arith.constant 0 : index
  affine.parallel (%t) = (0) to (symbol(%ni)) {
    %v = affine.load %in[%t] : memref<?xf64, 1>
    affine.store %v, %out[%t] : memref<?xf64, 1>
    "enzymexla.barrier"(%t, %c0, %c0) : (index, index, index) -> ()
    %w = affine.load %out[%t] : memref<?xf64, 1>
    affine.store %w, %in[%t] : memref<?xf64, 1>
  }
  return
}

// CHECK: barrier over a dynamically sized parallel axis
