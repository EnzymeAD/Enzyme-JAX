// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A loop that accumulates from 0.0, parallelized: the sum is the reduction
// alone, with no 0.0 added to it.
func.func @accumulate(%n: index, %x: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%s = %cst) -> (f64) {
    %v = affine.load %x[%i] : memref<?xf64>
    %t = arith.addf %s, %v : f64
    affine.yield %t : f64
  }
  return %r : f64
}

// CHECK:  func.func @accumulate(%arg0: index, %arg1: memref<?xf64>) -> f64 {
// CHECK-NEXT:   %0 = affine.parallel (%arg2) = (0) to (symbol(%arg0)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:     affine.yield %1 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }

// A sum from 1.0 keeps its addition of 1.0.
func.func @from_one(%n: index, %x: memref<?xf64>) -> f64 {
  %one = arith.constant 1.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%s = %one) -> (f64) {
    %v = affine.load %x[%i] : memref<?xf64>
    %t = arith.addf %s, %v : f64
    affine.yield %t : f64
  }
  return %r : f64
}

// CHECK:  func.func @from_one(%arg0: index, %arg1: memref<?xf64>) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.parallel (%arg2) = (0) to (symbol(%arg0)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %2 = affine.load %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = arith.addf %0, %cst : f64
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }

// A product can be -0.0, to which adding +0.0 gives +0.0: kept.
func.func @product(%n: index, %x: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.0 : f64
  %p = affine.parallel (%i) = (0) to (symbol(%n)) reduce ("mulf") -> (f64) {
    %v = affine.load %x[%i] : memref<?xf64>
    affine.yield %v : f64
  }
  %r = arith.addf %p, %cst : f64
  return %r : f64
}

// CHECK:  func.func @product(%arg0: index, %arg1: memref<?xf64>) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.parallel (%arg2) = (0) to (symbol(%arg0)) reduce ("mulf") -> (f64) {
// CHECK-NEXT:     %2 = affine.load %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = arith.addf %0, %cst : f64
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }
