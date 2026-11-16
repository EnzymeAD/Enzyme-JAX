// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A rotated loop sits in a check that it runs (`if (n > 0) do ... while`):
// where the check fails the loop runs no iteration, and the else branch
// yields what the loop would then, its initial value. The loop moves out of
// the check, and the check goes.
#runs = affine_set<()[s0] : (s0 - 1 >= 0)>
func.func @serial_rotated(%x: memref<?xf64>, %n: index) -> f64 {
  %z = arith.constant 0.0 : f64
  %v = affine.if #runs()[%n] -> f64 {
    %r = affine.for %i = 0 to %n iter_args(%s = %z) -> (f64) {
      %a = affine.load %x[%i] : memref<?xf64>
      %t = arith.addf %s, %a : f64
      affine.store %t, %x[%i + 1] : memref<?xf64>
      affine.yield %t : f64
    }
    affine.yield %r : f64
  } else {
    affine.yield %z : f64
  }
  return %v : f64
}

// CHECK:  func.func @serial_rotated(%arg0: memref<?xf64>, %arg1: index) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg2 = 0 to %arg1 iter_args(%arg3 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:     %2 = arith.addf %arg3, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg0[%arg2 + 1] : memref<?xf64>
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }

// A reducing parallel loop with no iteration reduces to the identity, which
// the then branch combines into the value the else branch yields.
func.func @parallel_rotated(%x: memref<?xf64>, %n: index, %init: f64) -> f64 {
  %v = affine.if #runs()[%n] -> f64 {
    %r = affine.parallel (%i) = (0) to (symbol(%n)) reduce ("addf") -> (f64) {
      %a = affine.load %x[%i] : memref<?xf64>
      affine.yield %a : f64
    }
    %t = arith.addf %r, %init : f64
    affine.yield %t : f64
  } else {
    affine.yield %init : f64
  }
  return %v : f64
}

// CHECK:  func.func @parallel_rotated(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) -> f64 {
// CHECK-NEXT:   %0 = affine.parallel (%arg3) = (0) to (symbol(%arg1)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %2 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = arith.addf %0, %arg2 : f64
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }

// A check that is not the loop's: the loop may run where it fails.
#other = affine_set<()[s0] : (s0 - 5 >= 0)>
func.func @other_check(%x: memref<?xf64>, %n: index, %k: index) -> f64 {
  %z = arith.constant 0.0 : f64
  %v = affine.if #other()[%k] -> f64 {
    %r = affine.for %i = 0 to %n iter_args(%s = %z) -> (f64) {
      %a = affine.load %x[%i] : memref<?xf64>
      %t = arith.addf %s, %a : f64
      affine.store %t, %x[%i + 1] : memref<?xf64>
      affine.yield %t : f64
    }
    affine.yield %r : f64
  } else {
    affine.yield %z : f64
  }
  return %v : f64
}

// CHECK:  func.func @other_check(%arg0: memref<?xf64>, %arg1: index, %arg2: index) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.if #set()[%arg2] -> f64 {
// CHECK-NEXT:     %1 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:       %2 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:       %3 = arith.addf %arg4, %2 : f64
// CHECK-NEXT:       affine.store %3, %arg0[%arg3 + 1] : memref<?xf64>
// CHECK-NEXT:       affine.yield %3 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %1 : f64
// CHECK-NEXT:   } else {
// CHECK-NEXT:     affine.yield %cst : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }
