// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The access reads a 3x3 block transposed, `3 * (3 * (i mod 3) + i floordiv 3
// + 3 * e)`, which isl writes with the factor outside and the affine
// simplifier multiplies back in. The pass converges, and the loop is parallel.
module {
  func.func @transposed(%x: memref<?xf64>, %y: memref<?xf64>, %e: index) {
    affine.for %i = 0 to 9 {
      %v = affine.load %x[(%i * 3 + (%i floordiv 3) * -8 + symbol(%e) * 3) * 3] : memref<?xf64>
      affine.store %v, %y[%i + symbol(%e) * 9] : memref<?xf64>
    }
    return
  }
}

// CHECK:  func.func @transposed(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (9) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3 * 9 + (%arg3 floordiv 3) * -24 + symbol(%arg2) * 9] : memref<?xf64>
// CHECK-NEXT:     affine.store %0, %arg1[%arg3 + symbol(%arg2) * 9] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
