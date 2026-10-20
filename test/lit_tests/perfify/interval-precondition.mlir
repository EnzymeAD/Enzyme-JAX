// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s

// Interval hypotheses on the starting cost (the precondition). Every
// function performs three arith.muli at cost 2 each; the precondition
// bounds the starting counter and the postcondition sees start + 6.

// starting cost in [0, 2], so the final cost (start + 6) is in [6, 8]: holds
module {
  func.func @pre_0_2_post_6_8(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @pre_0_2_post_6_8 true pre {
      %b0 = perfify.arg 0
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 2 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %b0, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 8 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @pre_0_2_post_6_8 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// starting cost in [0, 2] allows a final cost of 8, outside [6, 7]: does not hold
module {
  func.func @pre_0_2_post_6_7(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @pre_0_2_post_6_7 true pre {
      %b0 = perfify.arg 0
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 2 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %b0, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 7 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @pre_0_2_post_6_7 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an empty precondition [5, 3] is unsatisfiable, so the postcondition holds vacuously
module {
  func.func @pre_5_3_empty(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @pre_5_3_empty true pre {
      %b0 = perfify.arg 0
      %lb = perfify.constant_cost {
        %c = arith.constant 5 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 3 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %b0, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 8 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @pre_5_3_empty true pre {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
