// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s

// perfify.sorry gives an op an unknown symbolic cost S >= 0 instead of a
// concrete cost region. Every function performs three arith.muli at cost 3
// each plus one sorried arith.addi, so the total cost is 9 + S.

// 9 + S ge 9: holds for every S >= 0
module {
  func.func @sorry_ge(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    %a4 = arith.addi %a3, %a0 : i64
    return %a4 : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 3 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.sorry "arith.addi"
    perfify.conditions @sorry_ge true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.estimated_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %c9 = perfify.estimated_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp ge, %cost, %c9
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// The sorried op must not be reported as having an unknown cost, and since
// the post-condition holds there is no counterexample to report.
// CHECK-NOT: unknown cost for operation arith.addi
// CHECK-NOT: Condition can be proven if sorried op is resolved
// CHECK-LABEL: perfify.conditions @sorry_ge true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 9 + S le 9: does not hold, since S is unbounded. This only fails if the
// sorried op's cost is counted in fn_cost.
module {
  func.func @sorry_le(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    %a4 = arith.addi %a3, %a0 : i64
    return %a4 : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 3 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.sorry "arith.addi"
    perfify.conditions @sorry_le true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.estimated_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %c9 = perfify.estimated_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp le, %cost, %c9
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-NOT: unknown cost for operation arith.addi
// CHECK: Condition can be proven if sorried op is resolved
// CHECK-LABEL: perfify.conditions @sorry_le true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
