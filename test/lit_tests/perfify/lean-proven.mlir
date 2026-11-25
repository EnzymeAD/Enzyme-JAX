// RUN: enzymexlamlir-opt %s -split-input-file | enzymexlamlir-opt -split-input-file | FileCheck %s --check-prefix=PARSE
// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s --check-prefix=ANALYSIS

// lean_proven on perfify.conditions prints as a keyword, after invalid_region
// when both are present.
module {
  func.func @f(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.conditions @f true invalid_region = "(>= sym_X 6)" lean_proven pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 5 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = true}
    }
    perfify.conditions @f true lean_proven pre {
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
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @f true pre {
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
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// PARSE-LABEL: perfify.conditions @f true invalid_region = "(>= sym_X 6)" lean_proven pre {
// PARSE:       perfify.conditions @f true lean_proven pre {
// PARSE:       perfify.conditions @f true pre {
// PARSE-NOT:   lean_proven
// PARSE:       // -----

// The analysis leaves a lean_proven condition's region as it is. The region is
// removed by lean-to-perfify when it sets lean_proven, not by the analysis.
// ANALYSIS-LABEL: perfify.conditions @f true invalid_region = "(>= sym_X 6)" lean_proven pre {

// -----

// arith.addi has no cost, so Z3 cannot bound @proven, but lean_proven makes
// its post hold. @caller then uses that bound: 2 * 5 <= 10.
module {
  func.func @proven(%a0: i64) -> i64 {
    %s = arith.addi %a0, %a0 : i64
    return %s : i64
  }
  func.func @caller(%a0: i64) -> i64 {
    %x = func.call @proven(%a0) : (i64) -> i64
    %y = func.call @proven(%x) : (i64) -> i64
    return %y : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @proven true lean_proven pre {
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
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @caller true pre {
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
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// ANALYSIS-LABEL: perfify.conditions @proven true lean_proven pre {
// ANALYSIS:         perfify.assume %{{.*}} {satres = true}
// ANALYSIS:       } post {
// ANALYSIS:         perfify.assume %{{.*}} {satres = true}
// ANALYSIS-LABEL: perfify.conditions @caller true pre {
// ANALYSIS:         perfify.assume %{{.*}} {satres = true}
// ANALYSIS:       } post {
// ANALYSIS:         perfify.assume %{{.*}} {satres = true}

// -----

// The same callee without lean_proven has no proven bound, so neither it nor
// @caller holds.
module {
  func.func @unproven(%a0: i64) -> i64 {
    %s = arith.addi %a0, %a0 : i64
    return %s : i64
  }
  func.func @caller(%a0: i64) -> i64 {
    %x = func.call @unproven(%a0) : (i64) -> i64
    %y = func.call @unproven(%x) : (i64) -> i64
    return %y : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @unproven true pre {
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
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @caller true pre {
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
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// ANALYSIS-LABEL: perfify.conditions @unproven true pre {
// ANALYSIS:         perfify.assume %{{.*}} {satres = true}
// ANALYSIS:       } post {
// ANALYSIS:         perfify.assume %{{.*}} {satres = false}
// ANALYSIS-LABEL: perfify.conditions @caller true pre {
// ANALYSIS:         perfify.assume %{{.*}} {satres = true}
// ANALYSIS:       } post {
// ANALYSIS:         perfify.assume %{{.*}} {satres = false}
