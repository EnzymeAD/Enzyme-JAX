// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file %s | FileCheck %s
// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s --check-prefix=CFG

// A max whose order the enclosing bounds decide collapses: under t >= 0,
// max(t + 2, 2) is t + 2.
func.func @parallel(%out: memref<?xi32>) {
  %c2_i32 = arith.constant 2 : i32
  affine.parallel (%t) = (0) to (8) {
    %ti = arith.index_castui %t : index to i32
    %a = arith.addi %ti, %c2_i32 : i32
    %m = arith.maxsi %a, %c2_i32 : i32
    affine.store %m, %out[%t] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func.func @parallel(
// CHECK-NOT: arith.maxsi
// CHECK: %[[a:.+]] = arith.addi
// CHECK: affine.store %[[a]],

// -----

// The integer set of an enclosing affine.if constrains the domain: under
// n >= 1 and m >= 1, max(m, 1) is m, whether or not any affine loop encloses
// the use.
func.func @ifguard(%n: i32, %m: i32, %out: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %ni = arith.index_cast %n : i32 to index
  %mi = arith.index_cast %m : i32 to index
  affine.if affine_set<()[s0, s1] : (s0 - 1 >= 0, s1 - 1 >= 0)>()[%ni, %mi] {
    %mx = arith.maxsi %m, %c1_i32 : i32
    memref.store %mx, %out[%c0] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func.func @ifguard(
// CHECK-SAME: %[[n:.+]]: i32, %[[m:.+]]: i32,
// CHECK-NOT: arith.maxsi
// CHECK: memref.store %[[m]],

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
// n <= 0 decides the min.
func.func @ifelse(%n: index, %out: memref<?xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    } else {
      %ni = arith.index_cast %n : index to i32
      %mn = arith.minsi %ni, %c0_i32 : i32
      affine.store %mn, %out[%i] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @ifelse(
// CHECK-NOT: arith.minsi
// CHECK: %[[ni:.+]] = arith.index_cast
// CHECK: affine.store %[[ni]],

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

// -----

// An unsigned min/max only folds when both sides are provably non-negative:
// a subtraction that can go negative keeps its maxui.
func.func @unsigned_guard(%out: memref<?xi32>) {
  %c4_i32 = arith.constant 4 : i32
  affine.parallel (%t) = (0) to (8) {
    %a = arith.index_castui %t : index to i32
    %d = arith.subi %a, %c4_i32 : i32
    %m = arith.maxui %d, %c4_i32 : i32
    affine.store %m, %out[%t] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func.func @unsigned_guard(
// CHECK: arith.maxui

// -----

// Within affine-cfg the fold shares a driver with the raising: a loop whose
// bound the fold makes affine raises, its body then sees the loop in its
// domain, and the comparison against the bound inside folds.
func.func @raised(%n: index, %out: memref<?xi1>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  affine.parallel (%t) = (0) to (symbol(%n)) {
    %ti = arith.index_castui %t : index to i32
    %ni = arith.index_castui %n : index to i32
    %ub = arith.maxsi %ni, %ti : i32
    scf.for %j = %c0_i32 to %ub step %c1_i32 : i32 {
      %ji = arith.index_castui %j : i32 to index
      %q = arith.cmpi slt, %j, %ni : i32
      memref.store %q, %out[%ji] : memref<?xi1>
    }
  }
  return
}

// CFG-LABEL: func.func @raised(
// CFG-NOT: arith.maxsi
// CFG-NOT: scf.for
// CFG-NOT: arith.cmpi
// CFG: %[[T:.+]] = arith.constant true
// CFG: affine.store %[[T]],
