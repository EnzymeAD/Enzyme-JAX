// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s
// Cost regions must stay self-contained under CSE/canonicalization (Reactant runs both before analysis).
// RUN: enzymexlamlir-opt %s -split-input-file -cse -canonicalize -simple-cycle-analysis | FileCheck %s

// Integer interval cost hypotheses, checked at their boundaries.
//
// Every function performs three arith.muli at cost 2 each (total 6);
// func.return is free. eq/ne test membership in [lb, ub]; the ordering
// predicates must hold against every point of the interval (lt: cost < lb,
// le: cost <= lb, gt: cost > ub, ge: cost >= ub). An empty side is
// unbounded.

// 6 in [0, 8]: holds
module {
  func.func @in_0_8(%a0: i64) -> i64 {
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
    perfify.conditions @in_0_8 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
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

// CHECK-LABEL: perfify.conditions @in_0_8 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 in [6, 6]: holds
module {
  func.func @in_6_6(%a0: i64) -> i64 {
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
    perfify.conditions @in_6_6 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_6_6 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 in [7, 8]: does not hold
module {
  func.func @in_7_8(%a0: i64) -> i64 {
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
    perfify.conditions @in_7_8 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 7 : i64
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

// CHECK-LABEL: perfify.conditions @in_7_8 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 in [0, 5]: does not hold
module {
  func.func @in_0_5(%a0: i64) -> i64 {
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
    perfify.conditions @in_0_5 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 5 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_0_5 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 in [, 6]: holds
module {
  func.func @in_inf_6(%a0: i64) -> i64 {
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
    perfify.conditions @in_inf_6 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_inf_6 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 in [7, ]: does not hold
module {
  func.func @in_7_inf(%a0: i64) -> i64 {
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
    perfify.conditions @in_7_inf true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 7 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, ]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_7_inf true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 not in [0, 5]: holds
module {
  func.func @not_in_0_5(%a0: i64) -> i64 {
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
    perfify.conditions @not_in_0_5 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 5 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp ne, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @not_in_0_5 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 not in [0, 8]: does not hold
module {
  func.func @not_in_0_8(%a0: i64) -> i64 {
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
    perfify.conditions @not_in_0_8 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 8 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp ne, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @not_in_0_8 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 lt [7, 9]: holds
module {
  func.func @lt_7_9(%a0: i64) -> i64 {
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
    perfify.conditions @lt_7_9 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 7 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp lt, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @lt_7_9 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 lt [6, 9]: does not hold
module {
  func.func @lt_6_9(%a0: i64) -> i64 {
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
    perfify.conditions @lt_6_9 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp lt, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @lt_6_9 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 le [6, 9]: holds
module {
  func.func @le_6_9(%a0: i64) -> i64 {
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
    perfify.conditions @le_6_9 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp le, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @le_6_9 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 gt [0, 6]: does not hold
module {
  func.func @gt_0_6(%a0: i64) -> i64 {
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
    perfify.conditions @gt_0_6 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp gt, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @gt_0_6 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 ge [0, 6]: holds
module {
  func.func @ge_0_6(%a0: i64) -> i64 {
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
    perfify.conditions @ge_0_6 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %lb = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp ge, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @ge_0_6 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 6 lt [, 9]: does not hold
module {
  func.func @lt_inf_9(%a0: i64) -> i64 {
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
    perfify.conditions @lt_inf_9 true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %ub = perfify.constant_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %iv = perfify.interval [, %ub]
      %cmp = perfify.cmp lt, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @lt_inf_9 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 6 le 5 (scalar bound): does not hold
module {
  func.func @le_5_scalar(%a0: i64) -> i64 {
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
    perfify.conditions @le_5_scalar true pre {
      %b0 = perfify.arg 0
      %c0 = perfify.constant_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp eq, %c0, %b0
      perfify.assume %cmp
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      %bound = perfify.constant_cost {
        %c = arith.constant 5 : i64
        perfify.yield %c : i64
      } : !perfify.cost
      %cmp = perfify.cmp le, %cost, %bound
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @le_5_scalar true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
