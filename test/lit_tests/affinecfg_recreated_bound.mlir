// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// The inner bound composes to n - i - 1. recreateExpr rebuilds that sum, and
// the tree it builds need not be the one the canonicalizer left: (-d0 + s0) - 1
// and -d0 + (s0 - 1) are the same map, print the same, and are not the same
// object. Comparing those to decide whether the bound changed left the pattern
// and the canonicalizer handing the loop back and forth, so the greedy driver
// never reached a fixpoint and the pass failed -- silently, since its only
// failure path is applyPatternsGreedily.

func.func @triangular(%n : index, %A : memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  scf.for %i = %c0 to %n step %c1 {
    %rest = arith.subi %n, %i : index
    %ub = arith.subi %rest, %c1 : index
    scf.for %j = %c0 to %ub step %c1 {
      memref.store %cst, %A[%j] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL:   func.func @triangular(
// CHECK-SAME:      %[[n:.*]]: index, %[[A:.*]]: memref<?xf64>) {
// CHECK:           %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           affine.for %[[i:.*]] = 0 to %[[n]] {
// CHECK:             affine.parallel (%[[j:.*]]) = (0) to (-%[[i]] + symbol(%[[n]]) - 1) {
// CHECK:               affine.store %[[cst]], %[[A]]{{\[}}%[[j]]] : memref<?xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
