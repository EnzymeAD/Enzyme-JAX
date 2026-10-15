// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A carried value whose result and argument are both unused is dropped, as
// scf.for's canonicalization does; one whose result is used stays, and so do
// the loop's attributes.
func.func @dead_carry(%x: memref<?xf64>, %n: index, %junk: f64) -> f64 {
  %z = arith.constant 0.0 : f64
  %r:3 = affine.for %i = 0 to %n iter_args(%s = %z, %d = %junk, %p = %z) -> (f64, f64, f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %t = arith.addf %s, %a : f64
    %u = arith.mulf %p, %a : f64
    affine.store %t, %x[%i + 1] : memref<?xf64>
    affine.yield %t, %t, %u : f64, f64, f64
  } {test.kept}
  return %r#2 : f64
}

// CHECK:  func.func @dead_carry(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0:2 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %cst, %arg5 = %cst) -> (f64, f64) {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     %2 = arith.addf %arg4, %1 : f64
// CHECK-NEXT:     %3 = arith.mulf %arg5, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg0[%arg3 + 1] : memref<?xf64>
// CHECK-NEXT:     affine.yield %2, %3 : f64, f64
// CHECK-NEXT:   } {test.kept}
// CHECK-NEXT:   return %0#1 : f64
// CHECK-NEXT: }
