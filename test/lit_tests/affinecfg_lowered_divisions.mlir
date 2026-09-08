// RUN: enzymexlamlir-opt %s --lower-affine --affine-cfg | FileCheck %s

// lower-affine expands floordiv, ceildiv and mod into a truncating division
// or remainder under a sign test; affine-cfg reads those shapes back, so a
// loop bound built from an outer induction variable survives the round trip.

func.func @bounds(%arg0: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.parallel (%i) = (0) to (5) {
    affine.parallel (%j) = (0) to ((-%i) floordiv 2 + 3) {
      affine.store %cst, %arg0[%j] : memref<?xf64>
    }
    affine.parallel (%k) = (0) to (%i ceildiv 2 + 1) {
      affine.store %cst, %arg0[%k + 8] : memref<?xf64>
    }
    affine.parallel (%l) = (0) to (%i mod 3 + 1) {
      affine.store %cst, %arg0[%l + 16] : memref<?xf64>
    }
  }
  return
}

// CHECK:    func.func @bounds(%arg0: memref<?xf64>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (5) {
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to ((-%arg1) floordiv 2 + 3) {
// CHECK-NEXT:        affine.store %cst, %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (%arg1 ceildiv 2 + 1) {
// CHECK-NEXT:        affine.store %cst, %arg0[%arg2 + 8] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (%arg1 mod 3 + 1) {
// CHECK-NEXT:        affine.store %cst, %arg0[%arg2 + 16] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
