// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// An accumulation in place, w[k] += x[k + dy * 5] over a loop of unknown
// trip count: the value is carried in and out of the loop, the load and
// the store guarded by the loop running at all, and the loop is a parallel
// reduction.
func.func @accumulate(%x: memref<?xf64>, %w: memref<25xf64>, %n: index, %k: index) {
  affine.for %dy = 0 to %n {
    %xv = affine.load %x[%k + %dy * 5] : memref<?xf64>
    %acc = affine.load %w[symbol(%k)] {alignment = 8 : i64} : memref<25xf64>
    %s = arith.addf %acc, %xv : f64
    affine.store %s, %w[symbol(%k)] {alignment = 8 : i64} : memref<25xf64>
  }
  return
}

// CHECK:  func.func @accumulate(%arg0: memref<?xf64>, %arg1: memref<25xf64>, %arg2: index, %arg3: index) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.if #set()[%arg2] -> f64 {
// CHECK-NEXT:     %3 = affine.load %arg1[symbol(%arg3)] {alignment = 8 : i64} : memref<25xf64>
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   } else {
// CHECK-NEXT:     affine.yield %cst : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = affine.parallel (%arg4) = (0) to (symbol(%arg2)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %3 = affine.load %arg0[%arg4 * 5 + symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:   affine.if #set()[%arg2] {
// CHECK-NEXT:     affine.store %2, %arg1[symbol(%arg3)] {alignment = 8 : i64} : memref<25xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// The scratch of a kernel, w[dx][ey] accumulated over dy under the parallel
// (dx, ey): the loads and the store name the variables in different orders.
// The guards fold under the enclosing loops, which run only where n >= 1.
func.func @kernel(%x: memref<?xf64>, %b: memref<?xf64>, %n: index) {
  %w = memref.alloca() : memref<25xf64>
  %cst = arith.constant 0.0 : f64
  affine.parallel (%dx, %ey) = (0, 0) to (symbol(%n), symbol(%n)) {
    affine.store %cst, %w[%dx * 5 + %ey] : memref<25xf64>
    affine.for %dy = 0 to %n {
      %bv = affine.load %b[%ey + %dy * 5] : memref<?xf64>
      %xv = affine.load %x[%dx + %dy * 5] : memref<?xf64>
      %p = arith.mulf %bv, %xv : f64
      %acc = affine.load %w[%dx * 5 + %ey] : memref<25xf64>
      %s = arith.addf %acc, %p : f64
      affine.store %s, %w[%dx * 5 + %ey] : memref<25xf64>
    }
  }
  affine.for %i = 0 to 25 {
    %v = affine.load %w[%i] : memref<25xf64>
    affine.store %v, %x[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @kernel(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %alloca = memref.alloca() : memref<25xf64>
// CHECK-NEXT:   affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%arg2), symbol(%arg2)) {
// CHECK-NEXT:     affine.store %cst, %alloca[%arg4 + %arg3 * 5] : memref<25xf64>
// CHECK-NEXT:     %0 = affine.load %alloca[%arg4 + %arg3 * 5] : memref<25xf64>
// CHECK-NEXT:     %1 = affine.parallel (%arg5) = (0) to (symbol(%arg2)) reduce ("addf") -> (f64) {
// CHECK-NEXT:       %3 = affine.load %arg1[%arg4 + %arg5 * 5] : memref<?xf64>
// CHECK-NEXT:       %4 = affine.load %arg0[%arg3 + %arg5 * 5] : memref<?xf64>
// CHECK-NEXT:       %5 = arith.mulf %3, %4 : f64
// CHECK-NEXT:       affine.yield %5 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:     affine.store %2, %alloca[%arg4 + %arg3 * 5] : memref<25xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (25) {
// CHECK-NEXT:     %0 = affine.load %alloca[%arg3] : memref<25xf64>
// CHECK-NEXT:     affine.store %0, %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// A value carried from one element to the next, a[i + 1] = a[i] + ..., is
// left to the loops of known trip count: the loop stays as it is.
func.func @carry(%a: memref<?xf64>, %n: index) {
  %cst = arith.constant 1.0 : f64
  affine.for %i = 0 to %n {
    %v = affine.load %a[%i] : memref<?xf64>
    %s = arith.addf %v, %cst : f64
    affine.store %s, %a[%i + 1] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @carry(%arg0: memref<?xf64>, %arg1: index) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   affine.for %arg2 = 0 to %arg1 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:     %1 = arith.addf %0, %cst : f64
// CHECK-NEXT:     affine.store %1, %arg0[%arg2 + 1] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// Two accumulators a loop keeps in one scratch, at w[a + 5b] and
// w[a + 5b + 25]: each store's only other writer in the loop writes a
// different element in every iteration, so both carry, as one reduction
// of two values.
func.func @two_accumulators(%x: memref<?xf64>, %G: memref<?xf64>, %n: index, %d: index) {
  %w = memref.alloca() : memref<50xf64>
  %cst = arith.constant 0.0 : f64
  affine.parallel (%a, %b) = (0, 0) to (symbol(%d), symbol(%d)) {
    affine.store %cst, %w[%a + %b * 5] : memref<50xf64>
    affine.store %cst, %w[%a + %b * 5 + 25] : memref<50xf64>
    affine.for %i = 0 to %n {
      %u = affine.load %x[%i + %a * symbol(%n)] : memref<?xf64>
      %g0 = affine.load %G[%i * 2 + %b * symbol(%n)] : memref<?xf64>
      %g1 = affine.load %G[%i * 2 + %b * symbol(%n) + 1] : memref<?xf64>
      %p0 = arith.mulf %u, %g0 : f64
      %acc0 = affine.load %w[%a + %b * 5] : memref<50xf64>
      %s0 = arith.addf %acc0, %p0 : f64
      affine.store %s0, %w[%a + %b * 5] : memref<50xf64>
      %p1 = arith.mulf %u, %g1 : f64
      %acc1 = affine.load %w[%a + %b * 5 + 25] : memref<50xf64>
      %s1 = arith.addf %acc1, %p1 : f64
      affine.store %s1, %w[%a + %b * 5 + 25] : memref<50xf64>
    }
  }
  return
}

// CHECK:  func.func @two_accumulators(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: index) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %alloca = memref.alloca() : memref<50xf64>
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), symbol(%arg3)) {
// CHECK-NEXT:     affine.store %cst, %alloca[%arg4 + %arg5 * 5] : memref<50xf64>
// CHECK-NEXT:     affine.store %cst, %alloca[%arg4 + %arg5 * 5 + 25] : memref<50xf64>
// CHECK-NEXT:     %0:2 = affine.if #set()[%arg2] -> (f64, f64) {
// CHECK-NEXT:       %4 = affine.load %alloca[%arg4 + %arg5 * 5] : memref<50xf64>
// CHECK-NEXT:       %5 = affine.load %alloca[%arg4 + %arg5 * 5 + 25] : memref<50xf64>
// CHECK-NEXT:       affine.yield %4, %5 : f64, f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.yield %cst, %cst : f64, f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %1:2 = affine.parallel (%arg6) = (0) to (symbol(%arg2)) reduce ("addf", "addf") -> (f64, f64) {
// CHECK-NEXT:       %4 = affine.load %arg0[%arg6 + %arg4 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       %5 = affine.load %arg1[%arg6 * 2 + %arg5 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       %6 = affine.load %arg1[%arg6 * 2 + %arg5 * symbol(%arg2) + 1] : memref<?xf64>
// CHECK-NEXT:       %7 = arith.mulf %4, %5 : f64
// CHECK-NEXT:       %8 = arith.mulf %4, %6 : f64
// CHECK-NEXT:       affine.yield %7, %8 : f64, f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.addf %0#0, %1#0 : f64
// CHECK-NEXT:     %3 = arith.addf %0#1, %1#1 : f64
// CHECK-NEXT:     affine.if #set()[%arg2] {
// CHECK-NEXT:       affine.store %3, %alloca[%arg4 + %arg5 * 5 + 25] : memref<50xf64>
// CHECK-NEXT:       affine.store %2, %alloca[%arg4 + %arg5 * 5] : memref<50xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
