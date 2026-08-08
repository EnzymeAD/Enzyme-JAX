// RUN: enzymexlamlir-opt %s --canonicalize-scf-for --split-input-file | FileCheck %s

// A do-while's results are the condition args of its final evaluation -- the
// one whose comparison fails -- so the before region has run once more than
// the comparison held. The extra iteration was only forced for impure before
// regions, as if purity made the run unobservable; but the values are what is
// observed. This loop is MFEM's Mesh::GetNumGeometries after LICM has hoisted
// the load: counting the set bits of %bits in [%lo, %hi) with a pure body.
// Converted with a trip count one short, a quadrilateral mesh counted zero
// geometries and every FiniteElementSpace refused the mesh as unfinalized.
//
// The count for lo=2, hi=4 is over g in {2, 3}: two iterations, not one.

func.func @popcount_range(%lo: i32, %hi: i32, %bits: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%g = %lo, %acc = %c0) : (i32, i32) -> (i32, i32) {
    %bit0 = arith.shrui %bits, %g : i32
    %bit = arith.andi %bit0, %c1 : i32
    %acc2 = arith.addi %bit, %acc : i32
    %g2 = arith.addi %g, %c1 : i32
    %cont = arith.cmpi ne, %g2, %hi : i32
    scf.condition(%cont) %g2, %acc2 : i32, i32
  } do {
  ^bb0(%g2: i32, %acc2: i32):
    scf.yield %g2, %acc2 : i32, i32
  }
  return %r#1 : i32
}

// The upper bound gets the do-while treatment: one past the comparison bound,
// clamped so at least one iteration runs.

// CHECK-LABEL: func.func @popcount_range
// CHECK:         %[[LB:.+]] = arith.addi %arg0, %c1_i32 : i32
// CHECK:         %[[MAX:.+]] = arith.maxsi %arg1, %[[LB]] : i32
// CHECK:         %[[UB:.+]] = arith.addi %[[MAX]], %c1_i32 : i32
// CHECK:         scf.for %{{.+}} = %[[LB]] to %[[UB]] step %c1_i32

// -----

// No value here is computed in the before region at all -- the condition args
// are pure passthroughs -- but the loop permutes two slots each iteration, so
// which evaluation the results come from still matters: for %n = 1 the answer
// is (%x, %y), taken at the failing evaluation after one swap-back. Only a
// value from outside the loop, or a slot that refills with itself, can do
// without the extra iteration -- even a slot advanced by a loop-invariant
// step observes the final evaluation, one step past the last the body saw.

func.func @swap(%n: i32, %x: i32, %y: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:3 = scf.while (%i = %c0, %a = %x, %b = %y) : (i32, i32, i32) -> (i32, i32, i32) {
    %cont = arith.cmpi slt, %i, %n : i32
    scf.condition(%cont) %i, %b, %a : i32, i32, i32
  } do {
  ^bb0(%i0: i32, %b2: i32, %a2: i32):
    %i2 = arith.addi %i0, %c1 : i32
    scf.yield %i2, %b2, %a2 : i32, i32, i32
  }
  return %r#1, %r#2 : i32, i32
}

// CHECK-LABEL: func.func @swap
// CHECK:         %[[MAX:.+]] = arith.maxsi %arg0, %{{.+}} : i32
// CHECK:         %[[UB:.+]] = arith.addi %[[MAX]], %{{.+}} : i32
// CHECK:         scf.for %{{.+}} to %[[UB]]

// -----

// The plainest shape of all: the condition forwards the slot itself and the
// body advances it by one. The result is the failing evaluation's value --
// count to 3 and the answer is 3, the value the comparison rejected, not 2,
// the last the body saw. An advancing slot is not exempt: the conversion
// yields the induction variable, so the result would come out one step short
// without the widened bound.

func.func @count(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %cont = arith.cmpi slt, %i, %n : i32
    scf.condition(%cont) %i : i32
  } do {
  ^bb0(%i0: i32):
    %i2 = arith.addi %i0, %c1 : i32
    scf.yield %i2 : i32
  }
  return %r : i32
}

// CHECK-LABEL: func.func @count
// CHECK:         %[[CLAMP:.+]] = arith.maxsi %arg0, %c0_i32 : i32
// CHECK:         %[[PAST:.+]] = arith.addi %[[CLAMP]], %c1_i32 : i32
// CHECK:         %[[FOR:.+]] = scf.for %[[IV:.+]] = %c0_i32 to %[[PAST]] step %c1_i32
// CHECK:           scf.yield %[[IV]] : i32
// CHECK:         return %[[FOR]] : i32
