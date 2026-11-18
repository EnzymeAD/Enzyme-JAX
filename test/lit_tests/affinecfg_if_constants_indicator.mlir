// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// A loop bound chosen by the component, D1D - (c == 2) for c in [0, 3),
// is an affine.if of constants over the component; written as the indicator
// (c floordiv 2), the bound is affine and the loop raises and parallelizes.
func.func @component_bound(%x: memref<?xf64>, %d1d: i32) {
  %c-1_i32 = arith.constant -1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f64
  affine.for %c = 0 to 3 {
    %adj = affine.if affine_set<(d0) : (d0 - 2 == 0)>(%c) -> i32 {
      affine.yield %c-1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %n = arith.addi %d1d, %adj : i32
    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      %cc = arith.muli %c, %c16 : index
      %idx = arith.addi %ii, %cc : index
      memref.store %cst, %x[%idx] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @component_bound(%arg0: memref<?xf64>, %arg1: i32) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   affine.for %arg2 = 0 to 3 {
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to (-(%arg2 floordiv 2) + symbol(%0)) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// The bound (c >= 1 ? n - 1 : n) over c in [0, 4): the indicator of c - 1
// over [-1, 2] is (c + 2) floordiv 3.
func.func @ge_bound(%x: memref<?xf64>, %n: i32) {
  %c-1_i32 = arith.constant -1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f64
  affine.for %c = 0 to 4 {
    %adj = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%c) -> i32 {
      affine.yield %c-1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %ub = arith.addi %n, %adj : i32
    scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      %cc = arith.muli %c, %c16 : index
      %idx = arith.addi %ii, %cc : index
      memref.store %cst, %x[%idx] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @ge_bound(%arg0: memref<?xf64>, %arg1: i32) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   affine.for %arg2 = 0 to 4 {
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to (-((%arg2 + 2) floordiv 3) + symbol(%0)) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// The arms as values a constant apart, (c == 0) ? D1D : D1D - 1: the else
// arm is the symbol, and the bound is symbol(D1D) - 1 + [c == 0].
func.func @value_arms(%x: memref<?xf64>, %d1d: i32) {
  %c-1_i32 = arith.constant -1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f64
  %dm1 = arith.addi %d1d, %c-1_i32 : i32
  affine.for %c = 0 to 3 {
    %n = affine.if affine_set<(d0) : (d0 == 0)>(%c) -> i32 {
      affine.yield %d1d : i32
    } else {
      affine.yield %dm1 : i32
    }
    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      %cc = arith.muli %c, %c16 : index
      %idx = arith.addi %ii, %cc : index
      memref.store %cst, %x[%idx] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @value_arms(%arg0: memref<?xf64>, %arg1: i32) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   affine.for %arg2 = 0 to 3 {
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to ((-%arg2) floordiv 2 + symbol(%0)) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg3 + %arg2 * 16] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// A conditional of constants no loop bound or guard reads stays a
// conditional, a set the domain simplifications read.
func.func @kept(%x: memref<?xi32>) {
  %c-1_i32 = arith.constant -1 : i32
  %c0_i32 = arith.constant 0 : i32
  affine.parallel (%c) = (0) to (3) {
    %adj = affine.if affine_set<(d0) : (d0 - 2 == 0)>(%c) -> i32 {
      affine.yield %c-1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    affine.store %adj, %x[%c] : memref<?xi32>
  }
  return
}

// CHECK:  func.func @kept(%arg0: memref<?xi32>) {
// CHECK-NEXT:   %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   affine.parallel (%arg1) = (0) to (3) {
// CHECK-NEXT:     %0 = affine.if #set(%arg1) -> i32 {
// CHECK-NEXT:       affine.yield %c-1_i32 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.yield %c0_i32 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %0, %arg0[%arg1] : memref<?xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
