// RUN: enzymexlamlir-opt %s -simple-cycle-analysis -simple-cycle-analysis | FileCheck %s

// A second run of the analysis reuses the trip count symbol that the first
// run defined for the dynamic loop, through perfify.loop_trip on the loop,
// rather than defining it again. 3 * trip <= 30 still fails when trip >= 11.
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

// CHECK:       perfify.define_symbol @dyn_for_linearloop_trip0 : i64
// CHECK-NOT:   perfify.define_symbol
// CHECK:       scf.for
// CHECK:       } {perfify.loop_trip = @dyn_for_linearloop_trip0}
// CHECK-LABEL: perfify.conditions @dyn_for_linear true invalid_region = "(>= sym_dyn_for_linearloop_trip0 11)" pre {
// CHECK:         perfify.assume %{{.*}} {satres = true}
// CHECK:       } post {
// CHECK:         perfify.assume %{{.*}} {satres = false}
