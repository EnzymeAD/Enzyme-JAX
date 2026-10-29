// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

// An access through a subindex of a buffer rebuilt on the buffer itself keeps
// the alignment it named: it is the same element.

module {
  func.func @sub(%m: memref<?x4xf64>, %i: index, %v: f64) -> f64 {
    %c1 = arith.constant 1 : index
    %s = "enzymexla.subindex"(%m, %i) : (memref<?x4xf64>, index) -> memref<4xf64>
    memref.store %v, %s[%c1] alignment(4) : memref<4xf64>
    %r = memref.load %s[%c1] alignment(4) : memref<4xf64>
    %t = affine.load %s[2] {alignment = 4 : i64} : memref<4xf64>
    affine.store %t, %s[3] {alignment = 4 : i64} : memref<4xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @sub(%arg0: memref<?x4xf64>, %arg1: index, %arg2: f64) -> f64 {
// CHECK-NEXT:   %c3 = arith.constant 3 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   memref.store %arg2, %arg0[%arg1, %c1] alignment(4) : memref<?x4xf64>
// CHECK-NEXT:   %0 = memref.load %arg0[%arg1, %c1] alignment(4) : memref<?x4xf64>
// CHECK-NEXT:   %1 = memref.load %arg0[%arg1, %c2] alignment(4) : memref<?x4xf64>
// CHECK-NEXT:   memref.store %1, %arg0[%arg1, %c3] alignment(4) : memref<?x4xf64>
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }
