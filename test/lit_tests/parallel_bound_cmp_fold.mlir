// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file %s | FileCheck %s
// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s --check-prefix=CFG

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

// Thread-id tests under constant-extent axes fold, unsigned ones included
// since a thread id is non-negative.
func.func @tid(%out: memref<?xi1>) {
  affine.parallel (%t1, %t2) = (0, 0) to (2, 2) {
    %a = arith.index_castui %t1 : index to i32
    %b = arith.index_castui %t2 : index to i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %x = arith.cmpi ugt, %a, %c1 : i32
    %y = arith.cmpi ult, %b, %c2 : i32
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

// The else region of a single-inequality set is its complement: there
// n <= 0, deciding a comparison against zero.
func.func @ifelse(%n: index, %flag: memref<?xi1>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    } else {
      %ni = arith.index_cast %n : index to i32
      %pos = arith.cmpi sgt, %ni, %c0_i32 : i32
      affine.store %pos, %flag[%i] : memref<?xi1>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifelse(
// CHECK-NOT: arith.cmpi
// CHECK: %[[F:.+]] = arith.constant false
// CHECK: affine.store %[[F]],

// -----

// The else region of a multi-constraint set is a disjunction and carries no
// usable constraint: nothing folds.
func.func @ifelse_keep(%n: index, %m: index, %flag: memref<?xi1>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0, s1] : (s0 - 1 >= 0, s1 - 1 >= 0)>()[%n, %m] {
    } else {
      %ni = arith.index_cast %n : index to i32
      %pos = arith.cmpi sgt, %ni, %c0_i32 : i32
      affine.store %pos, %flag[%i] : memref<?xi1>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifelse_keep(
// CHECK: arith.cmpi sgt

// -----

// Within affine-cfg the fold shares a driver with the raising: a loop the
// raising makes affine puts its body in the domain, and the comparison
// against its bound inside folds.
func.func @raised(%n: index, %out: memref<?xi1>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  affine.parallel (%t) = (0) to (symbol(%n)) {
    %ni = arith.index_castui %n : index to i32
    scf.for %j = %c0_i32 to %ni step %c1_i32 : i32 {
      %ji = arith.index_castui %j : i32 to index
      %q = arith.cmpi slt, %j, %ni : i32
      memref.store %q, %out[%ji] : memref<?xi1>
    }
  }
  return
}

// CFG-LABEL: func.func @raised(
// CFG-NOT: arith.cmpi
// CFG: %[[T:.+]] = arith.constant true
// CFG: affine.store %[[T]],
