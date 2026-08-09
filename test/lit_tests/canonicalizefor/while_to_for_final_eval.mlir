// RUN: enzymexlamlir-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

// A while with an extra condition can exit either early (the extra condition
// goes false) or by exhausting the trip count. In the latter case the while
// still evaluates the before region one final time -- the evaluation whose
// comparison fails -- and it is that evaluation's condition args which become
// the loop results. With %p true the loop below runs the body for i = 0, 1, 2
// and then exits on the evaluation with i = 3, yielding 4.
//
// So the for must run one iteration past the bound to compute those results,
// while the after region stays guarded by the original bound so that it does
// not execute an extra time.

func.func @trip_count_exit(%p: i1) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c3 = arith.constant 3 : i64
  %r = scf.while (%i = %c0) : (i64) -> i64 {
    %cmp = arith.cmpi slt, %i, %c3 : i64
    %next = arith.addi %i, %c1 : i64
    %c = arith.andi %cmp, %p : i1
    scf.condition(%c) %next : i64
  } do {
  ^bb0(%a: i64):
    scf.yield %a : i64
  }
  return %r : i64
}

// CHECK-LABEL:   func.func @trip_count_exit(
// CHECK-SAME:                               %[[P:.*]]: i1) -> i64 {
// CHECK-NEXT:      %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[POISON:.*]] = ub.poison : i64
// CHECK-NEXT:      %[[TRUE:.*]] = arith.constant true
// The trip count is one past the bound, so that the final evaluation of the
// before region -- the one producing the results -- is performed.
// CHECK-NEXT:      %[[C4:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[FOR:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[CUR:.*]] = %[[C0]], %[[RES:.*]] = %[[POISON]], %[[FLAG:.*]] = %[[TRUE]]) -> (i64, i64, i1)  : i64 {
// Once the loop is done the before region is no longer evaluated and the
// results stay frozen at the evaluation which ended it.
// CHECK-NEXT:        %[[BEFORE:.*]]:2 = scf.if %[[FLAG]] -> (i64, i1) {
// CHECK-NEXT:          %[[NEXT:.*]] = arith.addi %[[CUR]], %[[C1]] : i64
// CHECK-NEXT:          scf.yield %[[NEXT]], %[[P]] : i64, i1
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[RES]], %[[FALSE]] : i64, i1
// CHECK-NEXT:        }
// The after region keeps the original bound, so it does not run an extra time.
// CHECK-NEXT:        %[[INBOUNDS:.*]] = arith.cmpi slt, %[[IV]], %[[C3]] : i64
// CHECK-NEXT:        %[[GUARD:.*]] = arith.andi %[[INBOUNDS]], %[[BEFORE]]#1 : i1
// CHECK-NEXT:        %[[AFTER:.*]] = scf.if %[[GUARD]] -> (i64) {
// CHECK-NEXT:          scf.yield %[[BEFORE]]#0 : i64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[POISON]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[AFTER]], %[[BEFORE]]#0, %[[BEFORE]]#1 : i64, i64, i1
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FOR]]#1 : i64
// CHECK-NEXT:    }

// -----

// Without an extra condition the loop can only exit by trip count, but the
// result is still the failing evaluation's view: counting 0, 1, 2 against a
// bound of 3 hands out 3, the value the comparison rejected, not 2, the last
// the body saw. The extra iteration is a bound one past the comparison's,
// clamped for the zero-trip case -- and with nothing else observable the
// loop then folds away: the result of a for yielding its own induction
// variable is the last executed IV in closed form, max(ub, 0) here.

func.func @no_extra_condition(%ub: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r = scf.while (%i = %c0) : (i64) -> i64 {
    %cmp = arith.cmpi slt, %i, %ub : i64
    scf.condition(%cmp) %i : i64
  } do {
  ^bb0(%a: i64):
    %next = arith.addi %a, %c1 : i64
    scf.yield %next : i64
  }
  return %r : i64
}

// CHECK-LABEL:   func.func @no_extra_condition(
// CHECK-SAME:                                  %[[UB:.*]]: i64) -> i64 {
// CHECK-NEXT:      %[[LB:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[CLAMP:.*]] = arith.maxsi %[[UB]], %[[LB]] : i64
// CHECK-NEXT:      %[[LAST:.*]] = arith.maxsi %[[CLAMP]], %[[LB]] : i64
// CHECK-NEXT:      return %[[LAST]] : i64
// CHECK-NEXT:    }

// -----

// The final evaluation of the before region is only observable through its side
// effects or through the loop results. The next three functions share the same
// loop and vary exactly one of those two properties at a time.
//
// Baseline: pure before region, result used, so the extra iteration is needed.

func.func @used_results(%p: i1, %ptr: memref<i64>) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c3 = arith.constant 3 : i64
  %r = scf.while (%i = %c0) : (i64) -> i64 {
    %cmp = arith.cmpi slt, %i, %c3 : i64
    %next = arith.addi %i, %c1 : i64
    %c = arith.andi %cmp, %p : i1
    scf.condition(%c) %next : i64
  } do {
  ^bb0(%a: i64):
    memref.store %a, %ptr[] : memref<i64>
    scf.yield %a : i64
  }
  return %r : i64
}

// CHECK-LABEL:   func.func @used_results(
// CHECK-SAME:                            %[[P:.*]]: i1,
// CHECK-SAME:                            %[[PTR:.*]]: memref<i64>) -> i64 {
// CHECK-NEXT:      %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[POISON:.*]] = ub.poison : i64
// CHECK-NEXT:      %[[TRUE:.*]] = arith.constant true
// The bound is extended by one step.
// CHECK-NEXT:      %[[C4:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[FOR:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[CUR:.*]] = %[[C0]], %[[RES:.*]] = %[[POISON]], %[[FLAG:.*]] = %[[TRUE]]) -> (i64, i64, i1)  : i64 {
// CHECK-NEXT:        %[[BEFORE:.*]]:2 = scf.if %[[FLAG]] -> (i64, i1) {
// CHECK-NEXT:          %[[NEXT:.*]] = arith.addi %[[CUR]], %[[C1]] : i64
// CHECK-NEXT:          scf.yield %[[NEXT]], %[[P]] : i64, i1
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[RES]], %[[FALSE]] : i64, i1
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[INBOUNDS:.*]] = arith.cmpi slt, %[[IV]], %[[C3]] : i64
// CHECK-NEXT:        %[[GUARD:.*]] = arith.andi %[[INBOUNDS]], %[[BEFORE]]#1 : i1
// CHECK-NEXT:        %[[AFTER:.*]] = scf.if %[[GUARD]] -> (i64) {
// CHECK-NEXT:          memref.store %[[BEFORE]]#0, %[[PTR]][] : memref<i64>
// CHECK-NEXT:          scf.yield %[[BEFORE]]#0 : i64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[POISON]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[AFTER]], %[[BEFORE]]#0, %[[BEFORE]]#1 : i64, i64, i1
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FOR]]#1 : i64
// CHECK-NEXT:    }

// -----

// Differs from @used_results only in that the result is unused, so the final
// evaluation is unobservable and no extra iteration is needed.

func.func @unused_results(%p: i1, %ptr: memref<i64>) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c3 = arith.constant 3 : i64
  %r = scf.while (%i = %c0) : (i64) -> i64 {
    %cmp = arith.cmpi slt, %i, %c3 : i64
    %next = arith.addi %i, %c1 : i64
    %c = arith.andi %cmp, %p : i1
    scf.condition(%c) %next : i64
  } do {
  ^bb0(%a: i64):
    memref.store %a, %ptr[] : memref<i64>
    scf.yield %a : i64
  }
  return
}

// CHECK-LABEL:   func.func @unused_results(
// CHECK-SAME:                              %[[P:.*]]: i1,
// CHECK-SAME:                              %[[PTR:.*]]: memref<i64>) {
// CHECK-NEXT:      %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[POISON:.*]] = ub.poison : i64
// CHECK-NEXT:      %[[TRUE:.*]] = arith.constant true
// The bound is not extended: no extra iteration.
// CHECK-NEXT:      %[[FOR:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[CUR:.*]] = %[[C0]], %[[RES:.*]] = %[[POISON]], %[[FLAG:.*]] = %[[TRUE]]) -> (i64, i64, i1)  : i64 {
// CHECK-NEXT:        %[[BEFORE:.*]]:2 = scf.if %[[FLAG]] -> (i64, i1) {
// CHECK-NEXT:          %[[NEXT:.*]] = arith.addi %[[CUR]], %[[C1]] : i64
// CHECK-NEXT:          scf.yield %[[NEXT]], %[[P]] : i64, i1
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[RES]], %[[FALSE]] : i64, i1
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[INBOUNDS:.*]] = arith.cmpi slt, %[[IV]], %[[C3]] : i64
// CHECK-NEXT:        %[[GUARD:.*]] = arith.andi %[[INBOUNDS]], %[[BEFORE]]#1 : i1
// CHECK-NEXT:        %[[AFTER:.*]] = scf.if %[[GUARD]] -> (i64) {
// CHECK-NEXT:          memref.store %[[BEFORE]]#0, %[[PTR]][] : memref<i64>
// CHECK-NEXT:          scf.yield %[[BEFORE]]#0 : i64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[POISON]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[AFTER]], %[[BEFORE]]#0, %[[BEFORE]]#1 : i64, i64, i1
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// -----

// Differs from @unused_results only in that the store sits in the before region
// rather than the after region. The final evaluation is now observable through
// that side effect, so the existing purity check forces the extra iteration even
// though the result is still unused.

func.func @impure_before(%p: i1, %ptr: memref<i64>) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c3 = arith.constant 3 : i64
  %r = scf.while (%i = %c0) : (i64) -> i64 {
    memref.store %i, %ptr[] : memref<i64>
    %cmp = arith.cmpi slt, %i, %c3 : i64
    %next = arith.addi %i, %c1 : i64
    %c = arith.andi %cmp, %p : i1
    scf.condition(%c) %next : i64
  } do {
  ^bb0(%a: i64):
    scf.yield %a : i64
  }
  return
}

// CHECK-LABEL:   func.func @impure_before(
// CHECK-SAME:                             %[[P:.*]]: i1,
// CHECK-SAME:                             %[[PTR:.*]]: memref<i64>) {
// CHECK-NEXT:      %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[POISON:.*]] = ub.poison : i64
// CHECK-NEXT:      %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:      %[[C4:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[FOR:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[CUR:.*]] = %[[C0]], %[[RES:.*]] = %[[POISON]], %[[FLAG:.*]] = %[[TRUE]]) -> (i64, i64, i1)  : i64 {
// CHECK-NEXT:        %[[BEFORE:.*]]:2 = scf.if %[[FLAG]] -> (i64, i1) {
// CHECK-NEXT:          memref.store %[[CUR]], %[[PTR]][] : memref<i64>
// CHECK-NEXT:          %[[NEXT:.*]] = arith.addi %[[CUR]], %[[C1]] : i64
// CHECK-NEXT:          scf.yield %[[NEXT]], %[[P]] : i64, i1
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[RES]], %[[FALSE]] : i64, i1
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[INBOUNDS:.*]] = arith.cmpi slt, %[[IV]], %[[C3]] : i64
// CHECK-NEXT:        %[[GUARD:.*]] = arith.andi %[[INBOUNDS]], %[[BEFORE]]#1 : i1
// CHECK-NEXT:        %[[AFTER:.*]] = scf.if %[[GUARD]] -> (i64) {
// CHECK-NEXT:          scf.yield %[[BEFORE]]#0 : i64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[POISON]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[AFTER]], %[[BEFORE]]#0, %[[BEFORE]]#1 : i64, i64, i1
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
