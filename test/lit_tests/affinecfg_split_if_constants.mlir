// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// An affine.if over a dimension yielding constants is a select no affine
// expression writes: a loop bounded by its result splits on the conditional,
// and each copy raises with the constant in place.
func.func @split_for(%d: i32, %out: memref<?xi32>, %v: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t, %c) = (0, 0) to (8, 2) {
    %e = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%c) -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %ub = arith.addi %d, %e : i32
    scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      memref.store %v, %out[%ii] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @split_for
// CHECK-SAME: %[[d:.+]]: i32,
// CHECK-DAG: %[[dm1:.+]] = arith.addi %[[d]], %{{.+}} : i32
// CHECK-DAG: %[[a:.+]] = arith.index_cast %[[dm1]]
// CHECK-DAG: %[[b:.+]] = arith.index_cast %[[d]]
// CHECK-NOT: scf.for
// CHECK: affine.if #{{.+}}(%{{.+}}) {
// CHECK-NEXT: affine.parallel (%{{.+}}) = (0) to (symbol(%[[a]]))
// CHECK: } else {
// CHECK-NEXT: affine.parallel (%{{.+}}) = (0) to (symbol(%[[b]]))

// -----

// A guard whose condition derives from the conditional splits the same way.
func.func @split_if(%d: i32, %out: memref<?xi32>, %v: i32) {
  %c0_i32 = arith.constant 0 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t, %c) = (0, 0) to (8, 2) {
    %e = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%c) -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %n = arith.addi %d, %e : i32
    %ti = arith.index_cast %t : index to i32
    %q = arith.cmpi slt, %ti, %n : i32
    scf.if %q {
      memref.store %v, %out[%t] : memref<?xi32>
    }
  }
  return
}

// CHECK-DAG: #[[S1:.+]] = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
// CHECK-DAG: #[[S2:.+]] = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>
// CHECK-LABEL: func.func @split_if
// CHECK-NOT: scf.if
// CHECK-NOT: arith.cmpi
// CHECK: affine.if #{{.+}}(%{{.+}}) {
// CHECK-NEXT: affine.if #[[S1]](
// CHECK: } else {
// CHECK-NEXT: affine.if #[[S2]](

// -----

// A conditional over symbols is a symbol itself and needs no split.
func.func @symbol_keep(%d: i32, %k: index, %out: memref<?xi32>, %v: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t) = (0) to (8) {
    %e = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%k] -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %ub = arith.addi %d, %e : i32
    scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      memref.store %v, %out[%ii] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @symbol_keep
// CHECK: affine.if #{{.+}}()[%{{.+}}] -> i32 {
// CHECK: affine.parallel
// CHECK-NOT: affine.if
