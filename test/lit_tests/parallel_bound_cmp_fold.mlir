// RUN: enzymexlamlir-opt --simplify-affine-exprs --split-input-file %s | FileCheck %s

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
