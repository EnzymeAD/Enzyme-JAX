// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s
// Cost regions must stay self-contained under CSE/canonicalization (Reactant runs both before analysis).
// RUN: enzymexlamlir-opt %s -split-input-file -cse -canonicalize -simple-cycle-analysis | FileCheck %s

// scf.for loops with a static trip count. The cost of every op in a loop body
// is scaled by the trip count. arith.muli costs 2; constants, the loop op
// itself, scf.yield and func.return are free.

// 4 iterations * 2 = 8 in [8, 8]: holds
module {
  func.func @loop_8(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a0) -> (i64) {
      %m = arith.muli %acc, %acc : i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @loop_8 true pre {
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
        %c = arith.constant 8 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @loop_8 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 4 iterations * 2 = 8 in [2, 2] (body counted once): does not hold
module {
  func.func @loop_not_2(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a0) -> (i64) {
      %m = arith.muli %acc, %acc : i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @loop_not_2 true pre {
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
        %c = arith.constant 2 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @loop_not_2 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// Nested 2 x 3 iterations * 2 = 12 in [12, 12]: holds
module {
  func.func @nested_12(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %a0) -> (i64) {
      %s = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc2 = %acc) -> (i64) {
        %m = arith.muli %acc2, %acc2 : i64
        scf.yield %m : i64
      }
      scf.yield %s : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @nested_12 true pre {
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
        %c = arith.constant 12 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @nested_12 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// Call in a loop: callee proven at [2, 2], 4 iterations = 8 in [8, 8]: holds
module {
  func.func @square(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  func.func @call_loop_8(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a0) -> (i64) {
      %m = func.call @square(%acc) : (i64) -> i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @square true pre {
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
        %c = arith.constant 2 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @call_loop_8 true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost2 {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost2
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost2 {
        %c = arith.constant 8 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range2 [min_cost: @post_cost2, max_cost: @post_cost2]
      %cmp = perfify.cmp eq, %cost, @post_range2
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @square true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @call_loop_8 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// Loops without a static trip count iterate a symbolic number of times: the
// analysis declares a fresh perfify.define_symbol @<func>loop_trip<k> for
// each, which no bound written beforehand can mention.

// Dynamic scf.for: trip * X in [N*X, N*X]: does not hold
module {
  func.func @dyn_for_fresh(%a0: i64, %n: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %m = arith.muli %acc, %acc : i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.define_symbol @N : i64
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @dyn_for_fresh true pre {
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
        %n = perfify.sym_cost @N : i64
        %x = perfify.sym_cost @X : i64
        %nx = arith.muli %n, %x : i64
        perfify.yield %nx : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK: perfify.define_symbol @dyn_for_freshloop_trip{{[0-9]+}} : i64
// CHECK-LABEL: perfify.conditions @dyn_for_fresh true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// scf.while with a free body; the only cost is the muli after the loop:
// 2 in [2, 2] for any trip count: holds
module {
  func.func @while_free_body(%a0: i64, %n: i64) -> (i64, i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %r:2 = scf.while (%i = %c0, %acc = %a0) : (i64, i64) -> (i64, i64) {
      %c = arith.cmpi slt, %i, %n : i64
      scf.condition(%c) %i, %acc : i64, i64
    } do {
    ^bb0(%i2: i64, %acc2: i64):
      %i3 = arith.addi %i2, %c1 : i64
      %acc3 = arith.addi %acc2, %i2 : i64
      scf.yield %i3, %acc3 : i64, i64
    }
    %m = arith.muli %r#1, %r#1 : i64
    return %r#0, %m : i64, i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.addi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.while" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.condition" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @while_free_body true pre {
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
        %c = arith.constant 2 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK: perfify.define_symbol @while_free_bodyloop_trip{{[0-9]+}} : i64
// CHECK-LABEL: perfify.conditions @while_free_body true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// scf.while with a muli in the body: 2 * trip in [0, 100] fails, since the
// trip count is unbounded
module {
  func.func @while_costly_body(%a0: i64, %n: i64) -> (i64, i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %r:2 = scf.while (%i = %c0, %acc = %a0) : (i64, i64) -> (i64, i64) {
      %c = arith.cmpi slt, %i, %n : i64
      scf.condition(%c) %i, %acc : i64, i64
    } do {
    ^bb0(%i2: i64, %acc2: i64):
      %m = arith.muli %acc2, %acc2 : i64
      %i3 = arith.addi %i2, %c1 : i64
      scf.yield %i3, %m : i64, i64
    }
    return %r#0, %r#1 : i64, i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.addi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.while" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.condition" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @while_costly_body true pre {
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
      perfify.estimated_cost @post_cost_min {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      perfify.estimated_cost @post_cost_max {
        %c = arith.constant 100 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost_min, max_cost: @post_cost_max]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK: perfify.define_symbol @while_costly_bodyloop_trip{{[0-9]+}} : i64
// CHECK-LABEL: perfify.conditions @while_costly_body true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// A perfify.hypothesis in a loop body constrains the loop's trip count; the
// perfify.arg it compares stands for the trip count. With trip <= N and a body
// costing 2: 2 * trip <= 2N holds, 2 * trip <= N does not.
module {
  func.func @while_trip_le_n(%a0: i64, %n: i64) -> (i64, i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %r:2 = scf.while (%i = %c0, %acc = %a0) : (i64, i64) -> (i64, i64) {
      %c = arith.cmpi slt, %i, %n : i64
      scf.condition(%c) %i, %acc : i64, i64
    } do {
    ^bb0(%i2: i64, %acc2: i64):
      %t = perfify.arg 0
      perfify.estimated_cost @trip_ub {
        %ub = perfify.sym_cost @N : i64
        perfify.yield %ub : i64
      }
      %tcmp = perfify.cmp le, %t, @trip_ub
      %th = perfify.hypothesis %tcmp
      %m = arith.muli %acc2, %acc2 : i64
      %i3 = arith.addi %i2, %c1 : i64
      scf.yield %i3, %m : i64, i64
    }
    return %r#0, %r#1 : i64, i64
  }
  perfify.define_symbol @N : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.addi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.while" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.condition" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @while_trip_le_n true pre {
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
        %n = perfify.sym_cost @N : i64
        %c2 = arith.constant 2 : i64
        %n2 = arith.muli %n, %c2 : i64
        perfify.yield %n2 : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @while_trip_le_n true pre {
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
        %n = perfify.sym_cost @N : i64
        perfify.yield %n : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK: perfify.define_symbol @while_trip_le_nloop_trip{{[0-9]+}} : i64
// CHECK-LABEL: perfify.conditions @while_trip_le_n true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @while_trip_le_n true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// trip < 100 with a body costing 2: 2 * trip <= 198 holds
module {
  func.func @while_trip_lt_100(%a0: i64, %n: i64) -> (i64, i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %r:2 = scf.while (%i = %c0, %acc = %a0) : (i64, i64) -> (i64, i64) {
      %c = arith.cmpi slt, %i, %n : i64
      scf.condition(%c) %i, %acc : i64, i64
    } do {
    ^bb0(%i2: i64, %acc2: i64):
      %t = perfify.arg 0
      perfify.estimated_cost @trip_ub {
        %ub = arith.constant 100 : i64
        perfify.yield %ub : i64
      }
      %tcmp = perfify.cmp lt, %t, @trip_ub
      %th = perfify.hypothesis %tcmp
      %m = arith.muli %acc2, %acc2 : i64
      %i3 = arith.addi %i2, %c1 : i64
      scf.yield %i3, %m : i64, i64
    }
    return %r#0, %r#1 : i64, i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.addi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.while" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.condition" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @while_trip_lt_100 true pre {
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
        %c = arith.constant 198 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK: perfify.define_symbol @while_trip_lt_100loop_trip{{[0-9]+}} : i64
// CHECK-LABEL: perfify.conditions @while_trip_lt_100 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// Dynamic scf.for with trip == N: trip * X in [N*X, N*X] holds
module {
  func.func @dyn_for_trip_eq_n(%a0: i64, %n: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %t = perfify.arg 0
      perfify.estimated_cost @trip {
        %tn = perfify.sym_cost @N : i64
        perfify.yield %tn : i64
      }
      %tcmp = perfify.cmp eq, %t, @trip
      %th = perfify.hypothesis %tcmp
      %m = arith.muli %acc, %acc : i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.define_symbol @N : i64
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @dyn_for_trip_eq_n true pre {
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
        %n = perfify.sym_cost @N : i64
        %x = perfify.sym_cost @X : i64
        %nx = arith.muli %n, %x : i64
        perfify.yield %nx : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK: perfify.define_symbol @dyn_for_trip_eq_nloop_trip{{[0-9]+}} : i64
// CHECK-LABEL: perfify.conditions @dyn_for_trip_eq_n true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// A static trip count wins over a loop hypothesis: 4 * 2 = 8 in [8, 8] holds
// although the hypothesis claims 5 iterations
module {
  func.func @static_ignores_hyp(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a0) -> (i64) {
      %t = perfify.arg 0
      perfify.estimated_cost @trip {
        %c5 = arith.constant 5 : i64
        perfify.yield %c5 : i64
      }
      %tcmp = perfify.cmp eq, %t, @trip
      %th = perfify.hypothesis %tcmp
      %m = arith.muli %acc, %acc : i64
      scf.yield %m : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @static_ignores_hyp true pre {
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
        %c = arith.constant 8 : i64
        perfify.yield %c : i64
      }
      perfify.interval @post_range [min_cost: @post_cost, max_cost: @post_cost]
      %cmp = perfify.cmp eq, %cost, @post_range
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-NOT: perfify.define_symbol @static_ignores_hyp
// CHECK-LABEL: perfify.conditions @static_ignores_hyp true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
