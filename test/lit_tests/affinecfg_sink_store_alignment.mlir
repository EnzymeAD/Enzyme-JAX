// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The one store that replaces a store in each arm of an if is of the same
// address as both, and keeps the alignment they named.

module {
  func.func @sink(%m: memref<?xf64>, %c: i1, %a: f64, %b: f64) {
    scf.if %c {
      affine.store %a, %m[2] {alignment = 4 : i64} : memref<?xf64>
    } else {
      affine.store %b, %m[2] {alignment = 4 : i64} : memref<?xf64>
    }
    return
  }
}

// CHECK-LABEL: func.func @sink(%arg0: memref<?xf64>, %arg1: i1, %arg2: f64, %arg3: f64) {
// CHECK-NEXT:   %0 = scf.if %arg1 -> (f64) {
// CHECK-NEXT:     scf.yield %arg2 : f64
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %arg3 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg0[2] {alignment = 4 : i64} : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }
