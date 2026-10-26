// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(affine-cfg)" --split-input-file | FileCheck %s

// A load whose index uses a conditional's results may be lifted into that
// conditional's branches only when reaching the conditional means reaching the
// load: here the load is additionally guarded, and the guard is what keeps the
// index in bounds, so lifting it evaluates an out-of-bounds address on the
// threads the guard excludes.

module {
  func.func @guarded(%m: memref<?xi32>, %out: memref<?xi32, 3>, %n: index, %s: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cm2_i32 = arith.constant -2 : i32
    affine.parallel (%x, %y) = (0, 0) to (symbol(%n), 4) {
      %xi = arith.index_castui %x : index to i32
      %yi = arith.index_castui %y : index to i32
      %lt2 = arith.cmpi ult, %yi, %c2_i32 : i32
      %ym2 = arith.addi %yi, %cm2_i32 overflow<nsw> : i32
      %a1 = arith.select %lt2, %yi, %ym2 : i32
      %or = arith.ori %a1, %xi : i32
      %g = arith.cmpi eq, %or, %c0_i32 : i32
      scf.if %g {
        %idx = arith.muli %s, %a1 overflow<nsw, nuw> : i32
        %ext = arith.extui %idx nneg : i32 to i64
        %ix = arith.index_cast %ext : i64 to index
        %v = memref.load %m[%ix] : memref<?xi32>
        %ox = arith.index_cast %a1 : i32 to index
        memref.store %v, %out[%ox] : memref<?xi32, 3>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @guarded(
// CHECK: %[[A1:.+]] = affine.if
// CHECK-NOT: load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: memref.store

// -----

// Without an intervening guard the lift is sound and must keep happening: the
// load ran unconditionally before, and runs once in whichever branch executes.

module {
  func.func @unguarded(%m: memref<?xi32>, %out: memref<?xi32, 3>, %s: i32) {
    %c2_i32 = arith.constant 2 : i32
    %cm2_i32 = arith.constant -2 : i32
    affine.parallel (%y) = (0) to (4) {
      %yi = arith.index_castui %y : index to i32
      %lt2 = arith.cmpi ult, %yi, %c2_i32 : i32
      %ym2 = arith.addi %yi, %cm2_i32 overflow<nsw> : i32
      %a1 = arith.select %lt2, %yi, %ym2 : i32
      %idx = arith.muli %s, %a1 overflow<nsw, nuw> : i32
      %ext = arith.extui %idx nneg : i32 to i64
      %ix = arith.index_cast %ext : i64 to index
      %v = memref.load %m[%ix] : memref<?xi32>
      %ox = arith.index_cast %a1 : i32 to index
      memref.store %v, %out[%ox] : memref<?xi32, 3>
    }
    return
  }
}

// CHECK-LABEL: func.func @unguarded(
// CHECK: affine.if
// CHECK: affine.load
// CHECK: affine.yield
// CHECK: } else {
// CHECK: affine.load
// CHECK: affine.yield

// -----

// The load's parent is a single-block loop, but the conditional sits outside
// it: the loop still decides whether and how often the load runs, so lifting
// the load beside the conditional is not unconditional execution either.

module {
  func.func @looped(%m: memref<?xi32>, %out: memref<?xi32, 3>, %n: index, %s: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cm2_i32 = arith.constant -2 : i32
    affine.parallel (%y) = (0) to (4) {
      %yi = arith.index_castui %y : index to i32
      %lt2 = arith.cmpi ult, %yi, %c2_i32 : i32
      %ym2 = arith.addi %yi, %cm2_i32 overflow<nsw> : i32
      %a1 = arith.select %lt2, %yi, %ym2 : i32
      scf.for %j = %c0 to %n step %c1 {
        %idx = arith.muli %s, %a1 overflow<nsw, nuw> : i32
        %ext = arith.extui %idx nneg : i32 to i64
        %ix = arith.index_cast %ext : i64 to index
        %v = memref.load %m[%ix] : memref<?xi32>
        %ox = arith.index_cast %a1 : i32 to index
        memref.store %v, %out[%ox] : memref<?xi32, 3>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @looped(
// CHECK: affine.if
// CHECK-NOT: load
// CHECK: affine.for
// CHECK: memref.load
// CHECK: memref.store

// -----

// Same shape, but the loop provably runs at least one iteration: reaching its
// parent block still means reaching the load, so the lift is sound and fires.

module {
  func.func @looped_const(%m: memref<?xi32>, %out: memref<?xi32, 3>, %s: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2_i32 = arith.constant 2 : i32
    %cm2_i32 = arith.constant -2 : i32
    affine.parallel (%y) = (0) to (4) {
      %yi = arith.index_castui %y : index to i32
      %lt2 = arith.cmpi ult, %yi, %c2_i32 : i32
      %ym2 = arith.addi %yi, %cm2_i32 overflow<nsw> : i32
      %a1 = arith.select %lt2, %yi, %ym2 : i32
      scf.for %j = %c0 to %c4 step %c1 {
        %idx = arith.muli %s, %a1 overflow<nsw, nuw> : i32
        %ext = arith.extui %idx nneg : i32 to i64
        %ix = arith.index_cast %ext : i64 to index
        %v = memref.load %m[%ix] : memref<?xi32>
        %ox = arith.index_cast %a1 : i32 to index
        memref.store %v, %out[%ox] : memref<?xi32, 3>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @looped_const(
// CHECK: affine.if
// CHECK: affine.load
// CHECK: affine.yield
// CHECK: } else {
// CHECK: affine.load
// CHECK: affine.yield
