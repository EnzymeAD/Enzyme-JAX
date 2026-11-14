// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s
// Cost regions must stay self-contained under CSE/canonicalization (Reactant runs both before analysis).
// RUN: enzymexlamlir-opt %s -split-input-file -cse -canonicalize -simple-cycle-analysis | FileCheck %s

// perfify.cost "*" is the cost of every op without a cost of its own, and
// perfify.cost "<dialect>.*" the cost of every such op in that dialect.
// Without either, such an op leaves the bound unproven (see @unknown_op in
// soundness.mlir).

// An op without a cost of its own takes its dialect's "<dialect>.*" cost, and
// only then "*". arith.muli costs 2, "arith.*" covers the arith.addi at 1 and
// "*" covers the func.return at 10, so @dialect_default costs exactly
// 2 + 2 + 1 + 10 = 15. If "arith.*" overrode arith.muli the cost would be 13,
// and if "*" overrode "arith.*" it would be 24, so either mistake changes
// which of the two bounds hold.
module {
  func.func @dialect_default(%a0: i64) -> i64 {
    %m1 = arith.muli %a0, %a0 : i64
    %m2 = arith.muli %m1, %m1 : i64
    %s = arith.addi %m2, %a0 : i64
    return %s : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.cost "*" {
      %c = arith.constant 10 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @dialect_default true pre {
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
        %c = arith.constant 15 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @dialect_default true pre {
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
        %c = arith.constant 14 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @dialect_default true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @dialect_default true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// "arith.*" doesn't cover other dialects: func.return has no cost, so the
// bound doesn't hold.
module {
  func.func @dialect_only(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.*" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @dialect_only true pre {
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
        %c = arith.constant 100 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @dialect_only true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
