// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

func.func @accumulate_with_disjoint_read(%A: memref<100x100xf64>, %B: memref<100x120xf64>) {
  affine.for %j = 0 to 120 {
    affine.for %k = 1 to 100 {
      %a = affine.load %A[%k, 0] : memref<100x100xf64>
      %b = affine.load %B[%k, %j] : memref<100x120xf64>
      %p = arith.mulf %a, %b : f64
      %acc = affine.load %B[0, %j] : memref<100x120xf64>
      %s = arith.addf %acc, %p : f64
      affine.store %s, %B[0, %j] : memref<100x120xf64>
    }
  }
  return
}

// CHECK:  func.func @accumulate_with_disjoint_read(%arg0: memref<100x100xf64>, %arg1: memref<100x120xf64>) {
// CHECK-NEXT:   affine.parallel (%arg2) = (0) to (120) {
// CHECK-NEXT:     %0 = affine.load %arg1[0, %arg2] : memref<100x120xf64>
// CHECK-NEXT:     %1 = affine.parallel (%arg3) = (1) to (100) reduce ("addf") -> (f64) {
// CHECK-NEXT:       %3 = affine.load %arg0[%arg3, 0] : memref<100x100xf64>
// CHECK-NEXT:       %4 = affine.load %arg1[%arg3, %arg2] : memref<100x120xf64>
// CHECK-NEXT:       %5 = arith.mulf %3, %4 : f64
// CHECK-NEXT:       affine.yield %5 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg1[0, %arg2] : memref<100x120xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
