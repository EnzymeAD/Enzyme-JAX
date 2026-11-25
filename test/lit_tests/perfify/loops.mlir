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

// CHECK-LABEL: perfify.conditions @loop_not_2 true {{.*}}pre {
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
// CHECK-LABEL: perfify.conditions @while_trip_le_n true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @while_trip_le_n true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// static trip counts repeat no symbol, so 'ne' is still exact: 8 ne 5 holds
module {
  func.func @ne_static(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %u = arith.index_cast %a0 : i64 to index
    %r = scf.for %o = %c0 to %c2 step %c1 iter_args(%v = %a0) -> (i64) {
      %x = scf.for %i = %c0 to %c2 step %c1 iter_args(%w = %v) -> (i64) {
        %m = arith.muli %w, %w : i64
        scf.yield %m : i64
      }
      scf.yield %x : i64
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
    perfify.cost "arith.index_cast" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
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
    perfify.conditions @ne_static true pre {
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
        %c = arith.constant 5 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ne, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @ne_static true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// a convex claim survives a repeated symbol: 4 <= cost <= 8, so le 8 holds
module {
  func.func @le_shared_trip(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %u = arith.index_cast %a0 : i64 to index
    %r = scf.for %o = %c0 to %c2 step %c1 iter_args(%v = %a0) -> (i64) {
      %x = scf.for %i = %c0 to %u step %c1 iter_args(%w = %v) -> (i64) {
        %t = perfify.arg 0
        perfify.estimated_cost @lo {
          %c = arith.constant 1 : i64
          perfify.yield %c : i64
        }
        perfify.estimated_cost @hi {
          %c = arith.constant 2 : i64
          perfify.yield %c : i64
        }
        perfify.interval @range [min_cost: @lo, max_cost: @hi]
        %tc = perfify.cmp eq, %t, @range
        %th = perfify.hypothesis %tc
        %m = arith.muli %w, %w : i64
        scf.yield %m : i64
      }
      scf.yield %x : i64
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
    perfify.cost "arith.index_cast" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
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
    perfify.conditions @le_shared_trip true pre {
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
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @le_shared_trip true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
