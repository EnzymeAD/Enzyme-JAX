// RUN: enzymexlamlir-opt %s --parallel-lower | FileCheck %s

// The memref accesses that replace a launch body's affine accesses keep the
// alignment they named.

module {
  func.func @launch(%m: memref<?xf64>, %v: f64, %n: index) {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %n, %gy = %n, %gz = %n) threads(%tx, %ty, %tz) in (%sx = %n, %sy = %n, %sz = %n) {
      affine.store %v, %m[%tx + 2] {alignment = 4 : i64} : memref<?xf64>
      %r = affine.load %m[%bx] {alignment = 4 : i64} : memref<?xf64>
      affine.store %r, %m[%tx] : memref<?xf64>
      gpu.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @launch(%arg0: memref<?xf64>, %arg1: f64, %arg2: index) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%arg2, %arg2, %arg2) step (%c1, %c1, %c1) {
// CHECK-NEXT:     scf.parallel (%arg6, %arg7, %arg8) = (%c0, %c0, %c0) to (%arg2, %arg2, %arg2) step (%c1, %c1, %c1) {
// CHECK-NEXT:       %0 = affine.apply #map(%arg6)
// CHECK-NEXT:       memref.store %arg1, %arg0[%0] alignment(4) : memref<?xf64>
// CHECK-NEXT:       %1 = memref.load %arg0[%arg3] alignment(4) : memref<?xf64>
// CHECK-NEXT:       memref.store %1, %arg0[%arg6] : memref<?xf64>
// CHECK-NEXT:       scf.reduce 
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.reduce 
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
