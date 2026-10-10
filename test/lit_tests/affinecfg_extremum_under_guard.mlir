// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The rotation of a loop that might run no times bounds it by max(n, 1),
// while the kernel runs it under `if (n > 0)`: there the bound is n, a
// symbol the loop raises over, where the max kept it an scf.for.
func.func @guarded(%n: i32, %X: memref<?xf64>, %Y: memref<?xf64>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  %pos = arith.cmpi sgt, %n, %c0_i32 : i32
  scf.if %pos {
    %nmax = arith.maxsi %n, %c1_i32 : i32
    %ub = arith.addi %nmax, %c1_i32 : i32
    scf.for %i = %c1_i32 to %ub step %c1_i32 : i32 {
      %im = arith.addi %i, %c1_i32 : i32
      %idx = arith.index_cast %im : i32 to index
      %v = memref.load %X[%idx] : memref<?xf64>
      %w = arith.addf %v, %cst : f64
      memref.store %w, %Y[%idx] : memref<?xf64>
    }
  }
  return
}

// The guard on the way is an `and` of comparisons, and the max is read
// through an index cast: still the bound.
func.func @conjunction(%n: i32, %m: i32, %X: memref<?xf64>, %Y: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  %pos = arith.cmpi sgt, %n, %c0_i32 : i32
  %posm = arith.cmpi sgt, %m, %c0_i32 : i32
  %both = arith.andi %pos, %posm : i1
  scf.if %both {
    %nmax = arith.maxsi %n, %c1_i32 : i32
    %ub = arith.index_cast %nmax : i32 to index
    scf.for %i = %c0 to %ub step %c1 {
      %v = memref.load %X[%i] : memref<?xf64>
      %w = arith.addf %v, %cst : f64
      memref.store %w, %Y[%i] : memref<?xf64>
    }
  }
  return
}

// A guard on another value, or one that gives less than the max asks
// (n >= 0 against max(n, 1)): the max stays.
func.func @unguarded(%n: i32, %m: i32, %X: memref<?xf64>, %Y: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  %posm = arith.cmpi sgt, %m, %c0_i32 : i32
  %nonneg = arith.cmpi sge, %n, %c0_i32 : i32
  %both = arith.andi %posm, %nonneg : i1
  scf.if %both {
    %nmax = arith.maxsi %n, %c1_i32 : i32
    %ub = arith.index_cast %nmax : i32 to index
    scf.for %i = %c0 to %ub step %c1 {
      %v = memref.load %X[%i] : memref<?xf64>
      %w = arith.addf %v, %cst : f64
      memref.store %w, %Y[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @guarded(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %1 = arith.cmpi sgt, %arg0, %c0_i32 : i32
// CHECK-NEXT:   scf.if %1 {
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to (symbol(%0)) {
// CHECK-NEXT:       %2 = affine.load %arg1[%arg3 + 2] : memref<?xf64>
// CHECK-NEXT:       %3 = arith.addf %2, %cst : f64
// CHECK-NEXT:       affine.store %3, %arg2[%arg3 + 2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @conjunction(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %1 = arith.cmpi sgt, %arg0, %c0_i32 : i32
// CHECK-NEXT:   %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// CHECK-NEXT:   %3 = arith.andi %1, %2 : i1
// CHECK-NEXT:   scf.if %3 {
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (symbol(%0)) {
// CHECK-NEXT:       %4 = affine.load %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:       %5 = arith.addf %4, %cst : f64
// CHECK-NEXT:       affine.store %5, %arg3[%arg4] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @unguarded(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %0 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:   %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// CHECK-NEXT:   %3 = arith.cmpi sge, %arg0, %c0_i32 : i32
// CHECK-NEXT:   %4 = arith.andi %2, %3 : i1
// CHECK-NEXT:   scf.if %4 {
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (symbol(%1)) {
// CHECK-NEXT:       %5 = affine.load %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:       %6 = arith.addf %5, %cst : f64
// CHECK-NEXT:       affine.store %6, %arg3[%arg4] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
