// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// An equality constraint is not implied by the same expression appearing as
// an inequality in an enclosing affine.if: s0 - 1 >= 0 admits every s0 >= 1,
// not only s0 == 1. Dropping it as redundant made the inner set vacuously
// true and replaced the conditional with its empty then-region -- in MFEM's
// batched LUSolve that deleted the whole back-substitution remainder (every
// solve returned its right-hand side unchanged).

#ge = affine_set<()[s0] : (s0 - 1 >= 0)>
#eq = affine_set<()[s0] : (s0 - 1 == 0)>
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
}

// CHECK: affine.if #set{{[0-9]*}}()[%{{.+}}] {
// CHECK: affine.if #set{{[0-9]*}}()[%{{.+}}] {
// CHECK-NEXT: } else {
// CHECK-NEXT: affine.store
