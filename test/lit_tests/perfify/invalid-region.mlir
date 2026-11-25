// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s

// When a post hypothesis does not hold, invalid_region on perfify.conditions
// is the set of symbol values (sym_*) for which it fails: the negated bound
// with every other Z3 variable existentially quantified away.

// Dynamic scf.for: 3 * trip <= 30 fails exactly when trip >= 11
module {
  func.func @dyn_for_linear(%a0: i64, %n: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %m = arith.muli %acc, %acc : i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 3 : i64
      perfify.yield %c : i64
    }
    perfify.cost "*" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @dyn_for_linear true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 30 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @dyn_for_linear true invalid_region = "(>= sym_dyn_for_linearloop_trip{{[0-9]+}} 11)" pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// With trip <= 10 assumed, 2 * trip + (trip + 1) <= 30 fails only at
// trip = 10. Without bound propagation the region is the pair of bounds
// (and (<= trip 10) (>= trip 10)) rather than the equality.
module {
  func.func @simplify_to_eq(%a0: tensor<i64>) -> tensor<i64> {
    %k = stablehlo.constant dense<1000> : tensor<i64>
    %r = stablehlo.while(%i = %a0) : tensor<i64>
    cond {
      %c = stablehlo.compare LT, %i, %k : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %t = perfify.arg 0
      perfify.estimated_cost @trip_ub {
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      %tcmp = perfify.cmp le, %t, @trip_ub
      %th = perfify.hypothesis %tcmp
      %s = stablehlo.multiply %i, %i : tensor<i64>
      stablehlo.return %s : tensor<i64>
    }
    return %r : tensor<i64>
  }
  perfify.assumptions {
    perfify.cost "stablehlo.multiply" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "stablehlo.compare" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.cost "*" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @simplify_to_eq true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 30 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @simplify_to_eq true invalid_region = "(= sym_simplify_to_eqloop_trip{{[0-9]+}} 10)" pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
