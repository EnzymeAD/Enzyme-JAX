// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// An equality constraint is not implied by the same expression appearing as
// an inequality in an enclosing affine.if: s0 - 1 >= 0 admits every s0 >= 1,
// not only s0 == 1. Dropping it as redundant made the inner set vacuously
// true and replaced the conditional with its empty then-region -- in MFEM's
// batched LUSolve that deleted the whole back-substitution remainder (every
// solve returned its right-hand side unchanged).

#ge = affine_set<()[s0] : (s0 - 1 >= 0)>
#eq = affine_set<()[s0] : (s0 - 1 == 0)>
#both = affine_set<()[s0] : (s0 - 1 >= 0, s0 - 1 == 0)>
module {
  func.func @f(%m: index, %out: memref<?xf64>) {
    %cst = arith.constant 1.0 : f64
    %cst2 = arith.constant 2.0 : f64
    affine.parallel (%q) = (0) to (8) {
      affine.if #ge()[%m] {
        affine.if #eq()[%m] {
        } else {
          affine.store %cst2, %out[%q] : memref<?xf64>
        }
      }
    }
    return
  }

  // The reverse direction within a single set does simplify: an inequality
  // conjoined with the same expression as an equality is the equality,
  // s0 - 1 >= 0 && s0 - 1 == 0 is s0 - 1 == 0.
  func.func @g(%m: index, %out: memref<?xf64>) {
    %cst2 = arith.constant 2.0 : f64
    affine.if #both()[%m] {
      affine.store %cst2, %out[0] : memref<?xf64>
    }
    return
  }
}

// CHECK-DAG: #[[GE:set[0-9]*]] = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK-DAG: #[[EQ:set[0-9]*]] = affine_set<()[s0] : (s0 - 1 == 0)>
// CHECK-LABEL: func.func @f
// CHECK: affine.if #[[GE]]()[%{{.+}}] {
// CHECK: affine.if #[[EQ]]()[%{{.+}}] {
// CHECK-NEXT: } else {
// CHECK-NEXT: affine.store
// CHECK-LABEL: func.func @g
// CHECK: affine.if #[[EQ]]()[%{{.+}}] {
// CHECK-NEXT: affine.store
