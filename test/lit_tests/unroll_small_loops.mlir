// RUN: enzymexlamlir-opt %s --unroll-small-loops=max_trip_count=3 --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --split-input-file --unroll-small-loops=max_trip_count=0 | FileCheck %s --check-prefix=OFF
// RUN: enzymexlamlir-opt %s --split-input-file --unroll-small-loops="max_trip_count=3 max_nested_ops=8" | FileCheck %s --check-prefix=BUDGET

// A loop over the three components sizing an inner loop by the component,
// `D1Dz = (c == 2) ? D1D : D1D - 1`: written out, each copy's bound is D1D
// or D1D - 1.
func.func @component(%x: memref<?xf64>, %d1d: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f64
  %d1dm1 = arith.subi %d1d, %c1 : index
  scf.for %c = %c0 to %c3 step %c1 {
    %isz = arith.cmpi eq, %c, %c2 : index
    %d1dz = arith.select %isz, %d1d, %d1dm1 : index
    scf.for %dz = %c0 to %d1dz step %c1 {
      memref.store %cst, %x[%dz] : memref<?xf64>
    }
  }
  return
}

// Four iterations is more than the maximum: the loop stays.
func.func @too_many(%x: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f64
  scf.for %i = %c0 to %c4 step %c1 {
    memref.store %cst, %x[%i] : memref<?xf64>
  }
  return
}

// An affine.for carrying a value: the copies chain it.
func.func @carried(%x: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to 2 iter_args(%acc = %cst) -> f64 {
    %v = affine.load %x[%i] : memref<?xf64>
    %s = arith.addf %acc, %v : f64
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK: func.func @component(%arg0: memref<?xf64>, %arg1: index) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.subi %arg1, %c1 : index
// CHECK-NEXT:   scf.for %arg2 = %c0 to %0 step %c1 {
// CHECK-NEXT:     memref.store %cst, %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.for %arg2 = %c0 to %0 step %c1 {
// CHECK-NEXT:     memref.store %cst, %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.for %arg2 = %c0 to %arg1 step %c1 {
// CHECK-NEXT:     memref.store %cst, %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @too_many(%arg0: memref<?xf64>) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   scf.for %arg1 = %c0 to %c4 step %c1 {
// CHECK-NEXT:     memref.store %cst, %arg0[%arg1] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @carried(%arg0: memref<?xf64>) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:   %1 = arith.addf %0, %cst : f64
// CHECK-NEXT:   %2 = affine.load %arg0[1] : memref<?xf64>
// CHECK-NEXT:   %3 = arith.addf %1, %2 : f64
// CHECK-NEXT:   return %3 : f64
// CHECK-NEXT: }

// -----

// Loops of no iteration and of a single one go whatever the maximum: the
// first is its initial value, the second its body. The two-iteration loop
// stays at a maximum of 0.
func.func @trivial(%x: memref<?xf64>, %init: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 1.0 : f64
  %none = scf.for %i = %c0 to %c0 step %c1 iter_args(%a = %init) -> f64 {
    %s = arith.addf %a, %cst : f64
    scf.yield %s : f64
  }
  %one = affine.for %i = 0 to 1 iter_args(%a = %none) -> f64 {
    %v = affine.load %x[%i] : memref<?xf64>
    %s = arith.addf %a, %v : f64
    affine.yield %s : f64
  }
  scf.for %i = %c0 to %c2 step %c1 {
    memref.store %one, %x[%i] : memref<?xf64>
  }
  return %one : f64
}

// CHECK: func.func @trivial(%arg0: memref<?xf64>, %arg1: f64) -> f64 {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:   %1 = arith.addf %arg1, %0 : f64
// CHECK-NEXT:   memref.store %1, %arg0[%c0] : memref<?xf64>
// CHECK-NEXT:   memref.store %1, %arg0[%c1] : memref<?xf64>
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }

// OFF: func.func @trivial(%arg0: memref<?xf64>, %arg1: f64) -> f64 {
// OFF-NEXT:   %c0 = arith.constant 0 : index
// OFF-NEXT:   %c0_0 = arith.constant 0 : index
// OFF-NEXT:   %c1 = arith.constant 1 : index
// OFF-NEXT:   %c2 = arith.constant 2 : index
// OFF-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// OFF-NEXT:   %0 = affine.load %arg0[%c0] : memref<?xf64>
// OFF-NEXT:   %1 = arith.addf %arg1, %0 : f64
// OFF-NEXT:   scf.for %arg2 = %c0_0 to %c2 step %c1 {
// OFF-NEXT:     memref.store %1, %arg0[%arg2] : memref<?xf64>
// OFF-NEXT:   }
// OFF-NEXT:   return %1 : f64
// OFF-NEXT: }

// -----

// A nest of small loops multiplies. With room for 8 operations, the inner
// loop (3 iterations of 2 operations) is written out; the outer one would
// leave 2 copies of those 6 and stays a loop.
func.func @budget(%x: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f64
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c3 step %c1 {
      %k = arith.addi %i, %j : index
      memref.store %cst, %x[%k] : memref<?xf64>
    }
  }
  return
}

// BUDGET: func.func @budget(%arg0: memref<?xf64>) {
// BUDGET-NEXT:   %c0 = arith.constant 0 : index
// BUDGET-NEXT:   %c1 = arith.constant 1 : index
// BUDGET-NEXT:   %c2 = arith.constant 2 : index
// BUDGET-NEXT:   %c3 = arith.constant 3 : index
// BUDGET-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// BUDGET-NEXT:   scf.for %arg1 = %c0 to %c2 step %c1 {
// BUDGET-NEXT:     %c3_0 = arith.constant 3 : index
// BUDGET-NEXT:     %0 = arith.addi %arg1, %c0 : index
// BUDGET-NEXT:     memref.store %cst, %arg0[%0] : memref<?xf64>
// BUDGET-NEXT:     %c1_1 = arith.constant 1 : index
// BUDGET-NEXT:     %1 = arith.muli %c1, %c1_1 : index
// BUDGET-NEXT:     %2 = arith.addi %c0, %1 : index
// BUDGET-NEXT:     %3 = arith.addi %arg1, %2 : index
// BUDGET-NEXT:     memref.store %cst, %arg0[%3] : memref<?xf64>
// BUDGET-NEXT:     %c2_2 = arith.constant 2 : index
// BUDGET-NEXT:     %4 = arith.muli %c1, %c2_2 : index
// BUDGET-NEXT:     %5 = arith.addi %c0, %4 : index
// BUDGET-NEXT:     %6 = arith.addi %arg1, %5 : index
// BUDGET-NEXT:     memref.store %cst, %arg0[%6] : memref<?xf64>
// BUDGET-NEXT:   }
// BUDGET-NEXT:   return
// BUDGET-NEXT: }
