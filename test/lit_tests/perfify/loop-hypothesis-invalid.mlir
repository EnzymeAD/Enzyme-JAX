// RUN: enzymexlamlir-opt %s -split-input-file -verify-diagnostics

// A hypothesis without pre/post must be in a loop body
perfify.estimated_cost @b {
  %c = arith.constant 0 : i64
  perfify.yield %c : i64
}
%t = perfify.arg 0
%cmp = perfify.cmp le, %t, @b
// expected-error @+1 {{must be in perfify.conditions, an scf.for body, or an scf.while 'after' region}}
%h = perfify.hypothesis %cmp

// -----

func.func @pre_in_loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = perfify.arg 0
    perfify.estimated_cost @b {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    %cmp = perfify.cmp le, %t, @b
    // expected-error @+1 {{loop hypothesis must not be 'pre' or 'post'}}
    %h = perfify.hypothesis %cmp pre
  }
  return
}

// -----

// The condition region runs once more than the trip count, so it may not hold
// the hypothesis
func.func @in_while_before(%n: i64) {
  %c0 = arith.constant 0 : i64
  scf.while (%i = %c0) : (i64) -> () {
    %t = perfify.arg 0
    perfify.estimated_cost @b {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    %cmp = perfify.cmp le, %t, @b
    // expected-error @+1 {{must be in perfify.conditions, an scf.for body, or an scf.while 'after' region}}
    %h = perfify.hypothesis %cmp
    %c = arith.cmpi slt, %i, %n : i64
    scf.condition(%c)
  } do {
    scf.yield %c0 : i64
  }
  return
}

// -----

func.func @not_arg(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = perfify.fn_cost : !perfify.cost
    perfify.estimated_cost @b {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    %cmp = perfify.cmp le, %t, @b
    // expected-error @+1 {{loop hypothesis must compare a perfify.arg}}
    %h = perfify.hypothesis %cmp
  }
  return
}
