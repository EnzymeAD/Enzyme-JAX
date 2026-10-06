// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The carried sum is added to, but the loop yields an affine.if over the sum:
// where the condition fails it starts over from another value. The add's one
// use is the affine.if's yield, not the loop's, so the carried value is no
// reduction and the loop stays serial.
#runs = affine_set<()[s0] : (s0 - 1 >= 0)>
func.func @reset_sum(%x: memref<?xf64>, %n: index, %m: index, %r: f64) -> f64 {
  %z = arith.constant 0.0 : f64
  %s = affine.for %i = 0 to %n iter_args(%a = %z) -> (f64) {
    %v = affine.load %x[%i] : memref<?xf64>
    %t = arith.addf %a, %v : f64
    %u = affine.if #runs()[%m] -> f64 {
      affine.yield %t : f64
    } else {
      affine.yield %r : f64
    }
    affine.yield %u : f64
  }
  return %s : f64
}

// CHECK:  func.func @reset_sum(%arg0: memref<?xf64>, %arg1: index, %arg2: index, %arg3: f64) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg4 = 0 to %arg1 iter_args(%arg5 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:     %2 = arith.addf %arg5, %1 : f64
// CHECK-NEXT:     %3 = affine.if #set()[%arg2] -> f64 {
// CHECK-NEXT:       affine.yield %2 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.yield %arg3 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }

// A sum the loop yields directly is a reduction, and the loop parallel.
func.func @sum(%x: memref<?xf64>, %n: index) -> f64 {
  %z = arith.constant 0.0 : f64
  %s = affine.for %i = 0 to %n iter_args(%a = %z) -> (f64) {
    %v = affine.load %x[%i] : memref<?xf64>
    %t = arith.addf %a, %v : f64
    affine.yield %t : f64
  }
  return %s : f64
}

// CHECK:  func.func @sum(%arg0: memref<?xf64>, %arg1: index) -> f64 {
// CHECK-NEXT:   %0 = affine.parallel (%arg2) = (0) to (symbol(%arg1)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:     affine.yield %1 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }
