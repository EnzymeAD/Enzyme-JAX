// RUN: enzymexlamlir-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

// A condition of two comparisons is tried one at a time on the same helper.
// The first finds an induction variable and is turned down later, on its
// predicate; the second finds none, and has to be left with none. What it was
// told about the first is not an answer to the second.

func.func @stale_indvar(%n: i32, %m: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %eq = arith.cmpi eq, %i, %n : i32
    %lt = arith.cmpi slt, %m, %n : i32
    %c = arith.andi %eq, %lt : i1
    scf.condition(%c) %i : i32
  } do {
  ^bb0(%a: i32):
    %next = arith.addi %a, %c1 : i32
    scf.yield %next : i32
  }
  return %r : i32
}

// Neither comparison gives a for, so the while stands.

// CHECK-LABEL: func.func @stale_indvar
// CHECK:         scf.while
// CHECK:         scf.condition
