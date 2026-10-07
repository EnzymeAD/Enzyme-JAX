// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A parallel loop variable used as its quotient and remainder by 3 is split
// into the two. In the second case it also bounds an inner loop, a use the
// split rewrites as an affine.apply of the two, a valid bound operand (the
// arith reconstruction was not, and the pass failed to verify).
module {
  func.func @split(%x: memref<?xf64>, %y: memref<?xf64>) {
    affine.parallel (%i) = (0) to (9) {
      %v = affine.load %x[(%i floordiv 3) * 16 + %i mod 3] : memref<?xf64>
      affine.store %v, %y[(%i floordiv 3) * 16 + %i mod 3] : memref<?xf64>
    }
    return
  }

  func.func @bound_use(%x: memref<?xf64>, %y: memref<?xf64>) {
    affine.parallel (%i) = (0) to (9) {
      %v = affine.load %x[(%i floordiv 3) * 16 + %i mod 3] : memref<?xf64>
      affine.for %j = 0 to affine_map<(d0) -> (d0 floordiv 3)>(%i) {
        affine.store %v, %y[%j + (%i floordiv 3) * 16 + %i mod 3] : memref<?xf64>
      }
    }
    return
  }
}



// CHECK:  func.func @split(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
// CHECK-NEXT:   affine.parallel (%arg2, %arg3) = (0, 0) to (3, 3) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:     affine.store %0, %arg1[%arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @bound_use(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
// CHECK-NEXT:   affine.parallel (%arg2, %arg3) = (0, 0) to (3, 3) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (%arg2 + %arg3 floordiv 3) {
// CHECK-NEXT:       affine.store %0, %arg1[%arg4 + %arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
