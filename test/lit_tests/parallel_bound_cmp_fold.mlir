// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file %s | FileCheck %s

// The residual of a peeled grid-stride loop guards on blockIdx + gridDim < N
// where the parallel extent and gridDim share a base: infeasible for any
// blockIdx >= 0, so the comparison folds to false.
func.func @gridstride(%n: index, %out: memref<?xi1>) {
  affine.parallel (%b) = (0) to (symbol(%n)) {
    %bi = arith.index_castui %b : index to i32
    %ni = arith.index_castui %n : index to i32
    %e1 = arith.addi %bi, %ni : i32
    %dead = arith.cmpi slt, %e1, %ni : i32
    affine.store %dead, %out[%b] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @gridstride(
// CHECK-NOT: arith.cmpi
// CHECK: %[[F:.+]] = arith.constant false
// CHECK: affine.store %[[F]],

// -----

// Thread-id tests under constant-extent axes fold, including a bitwise-or
// chain under a power-of-two bound (or of non-negative values stays under
// 2^k exactly when every operand does).
func.func @tid(%out: memref<?xi1>) {
  affine.parallel (%t1, %t2) = (0, 0) to (2, 2) {
    %a = arith.index_castui %t1 : index to i32
    %b = arith.index_castui %t2 : index to i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %x = arith.cmpi ugt, %a, %c1 : i32
    %o = arith.ori %a, %b : i32
    %y = arith.cmpi ult, %o, %c2 : i32
    %z = arith.andi %x, %y : i1
    affine.store %z, %out[%t1 * 2 + %t2] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @tid(
// CHECK-NOT: arith.cmpi

// -----

// A comparison against an unrelated symbol is not decided by the bounds and
// must survive.
func.func @keep(%n: index, %m: index, %out: memref<?xi1>) {
  affine.parallel (%b) = (0) to (symbol(%n)) {
    %bi = arith.index_castui %b : index to i32
    %mi = arith.index_castui %m : index to i32
    %q = arith.cmpi slt, %bi, %mi : i32
    affine.store %q, %out[%b] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @keep(
// CHECK: arith.cmpi slt

// -----

// Unsigned predicates only fold when both sides are provably non-negative on
// the domain: a subtraction that can go negative must survive `ult`.
func.func @unsigned_guard(%out: memref<?xi1>) {
  affine.parallel (%t) = (0) to (8) {
    %a = arith.index_castui %t : index to i32
    %c4 = arith.constant 4 : i32
    %d = arith.subi %a, %c4 : i32
    %q = arith.cmpi ult, %d, %c4 : i32
    affine.store %q, %out[%t] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @unsigned_guard(
// CHECK: arith.cmpi ult

// -----

// A strided remainder loop whose extent the bounds decide: maxsi collapses,
// the trip count proves to be exactly one, and the body inlines at the lower
// bound.
func.func @remainder(%out: memref<?xf64>, %v: f64) {
  %c2_i32 = arith.constant 2 : i32
  %c2 = arith.constant 2 : index
  affine.parallel (%t) = (0) to (2) {
    %t2 = arith.addi %t, %c2 : index
    %lb = arith.index_castui %t2 : index to i32
    %m = arith.maxsi %lb, %c2_i32 : i32
    %ub = arith.addi %m, %c2_i32 : i32
    scf.for %j = %lb to %ub step %c2_i32 : i32 {
      %ji = arith.index_castui %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @remainder(
// CHECK-NOT: arith.maxsi
// CHECK-NOT: scf.for
// CHECK: memref.store

// -----

// A rotated do-while remainder whose condition is false on its first
// evaluation is one execution of its before region.
func.func @dowhile(%out: memref<?xf64>, %v: f64) {
  %c2_i32 = arith.constant 2 : i32
  %c2 = arith.constant 2 : index
  affine.parallel (%t) = (0) to (2) {
    %t2 = arith.addi %t, %c2 : index
    %init = arith.index_castui %t2 : index to i32
    scf.while (%j = %init) : (i32) -> () {
      %ji = arith.index_castui %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
      %cond = arith.cmpi ult, %j, %c2_i32 : i32
      scf.condition(%cond)
    } do {
      %next = arith.addi %init, %c2_i32 : i32
      scf.yield %next : i32
    }
  }
  return
}

// CHECK-LABEL: func.func @dowhile(
// CHECK-NOT: scf.while
// CHECK: memref.store

// -----

// A provably zero-trip loop folds to its inits.
func.func @zerotrip(%n: index, %out: memref<?xf64>, %v: f64) -> f64 {
  %c1_i32 = arith.constant 1 : i32
  %r = affine.parallel (%b) = (0) to (symbol(%n)) reduce ("addf") -> f64 {
    %bi = arith.index_castui %b : index to i32
    %ni = arith.index_castui %n : index to i32
    %s = scf.for %j = %ni to %bi step %c1_i32 iter_args(%acc = %v) -> f64 : i32 {
      %a = arith.addf %acc, %acc : f64
      scf.yield %a : f64
    }
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK-LABEL: func.func @zerotrip(
// CHECK-NOT: scf.for
// CHECK: affine.yield %arg2

// -----

// The integer set of an enclosing affine.if constrains the domain: under
// n >= 1 and m >= 1 a rotated do-while's max(m, 1) is m, whether or not any
// affine loop encloses the use.
func.func @ifguard(%n: i32, %m: i32, %out: memref<?xf64>, %v: f64) {
  %c1_i32 = arith.constant 1 : i32
  %ni = arith.index_cast %n : i32 to index
  %mi = arith.index_cast %m : i32 to index
  affine.if affine_set<()[s0, s1] : (s0 - 1 >= 0, s1 - 1 >= 0)>()[%ni, %mi] {
    %mx = arith.maxsi %m, %c1_i32 : i32
    %ub = arith.addi %mx, %c1_i32 : i32
    scf.for %j = %c1_i32 to %ub step %c1_i32 : i32 {
      %ji = arith.index_cast %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifguard(
// CHECK-SAME: %[[n:.+]]: i32, %[[m:.+]]: i32,
// CHECK-NOT: arith.maxsi
// CHECK: %[[ub:.+]] = arith.addi %[[m]], %{{.+}} : i32
// CHECK: scf.for %{{.+}} = %{{.+}} to %[[ub]]

// -----

// Constraints combine with the enclosing loop bounds: inside i < n the
// remaining extent n - i is at least one.
func.func @ifiv(%n: index, %out: memref<?xi32>) {
  %c1_i32 = arith.constant 1 : i32
  affine.for %i = 0 to %n {
    affine.if affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>(%i)[%n] {
      %ni = arith.index_cast %n : index to i32
      %ii = arith.index_cast %i : index to i32
      %rem = arith.subi %ni, %ii : i32
      %mx = arith.maxsi %rem, %c1_i32 : i32
      affine.store %mx, %out[%i] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifiv(
// CHECK-NOT: arith.maxsi
// CHECK: %[[rem:.+]] = arith.subi
// CHECK: affine.store %[[rem]],

// -----

// The else region of a single-inequality set is its complement: there
// n <= 0, deciding both the min and a comparison against zero.
func.func @ifelse(%n: index, %out: memref<?xi32>, %flag: memref<?xi1>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    } else {
      %ni = arith.index_cast %n : index to i32
      %mn = arith.minsi %ni, %c0_i32 : i32
      %pos = arith.cmpi sgt, %ni, %c0_i32 : i32
      affine.store %mn, %out[%i] : memref<?xi32>
      affine.store %pos, %flag[%i] : memref<?xi1>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifelse(
// CHECK-NOT: arith.minsi
// CHECK-NOT: arith.cmpi
// CHECK-DAG: %[[ni:.+]] = arith.index_cast
// CHECK-DAG: %[[F:.+]] = arith.constant false
// CHECK: affine.store %[[ni]],
// CHECK: affine.store %[[F]],

// -----

// The else region of a multi-constraint set is a disjunction and carries no
// usable constraint: nothing folds.
func.func @ifelse_keep(%n: index, %m: index, %out: memref<?xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0, s1] : (s0 - 1 >= 0, s1 - 1 >= 0)>()[%n, %m] {
    } else {
      %ni = arith.index_cast %n : index to i32
      %mn = arith.minsi %ni, %c0_i32 : i32
      affine.store %mn, %out[%i] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifelse_keep(
// CHECK: arith.minsi
