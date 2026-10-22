// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s
// Cost regions must stay self-contained under CSE/canonicalization (Reactant runs both before analysis).
// RUN: enzymexlamlir-opt %s -split-input-file -cse -canonicalize -simple-cycle-analysis | FileCheck %s

// Symbolic interval cost hypotheses. Every function performs three
// arith.muli at cost X each (total 3X); func.return is free. A hypothesis
// holds only if it holds for every X >= 0 (symbolic costs are never
// negative), so e.g. [0, 8] does not hold since X is unbounded.

// 3X in [X, 3X]: holds
module {
  func.func @in_1x_3x(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @in_1x_3x true pre {
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
        %x = perfify.sym_cost @X : i64
        perfify.yield %x : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_1x_3x true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 3X in [3X, 3X]: holds
module {
  func.func @in_3x_3x(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @in_3x_3x true pre {
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
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_3x_3x true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 3X in [0, 3X+1]: holds
module {
  func.func @in_0_3xp1(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @in_0_3xp1 true pre {
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
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        %c = arith.constant 1 : i64
        %s = arith.addi %kx, %c : i64
        perfify.yield %s : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_0_3xp1 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 3X in [, 3X]: holds
module {
  func.func @in_inf_3x(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @in_inf_3x true pre {
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
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %iv = perfify.interval [, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_inf_3x true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// 3X in [X, 2X]: does not hold
module {
  func.func @in_1x_2x(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @in_1x_2x true pre {
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
        %x = perfify.sym_cost @X : i64
        perfify.yield %x : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 2 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_1x_2x true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 3X in [3X+1, 4X]: does not hold
module {
  func.func @in_3xp1_4x(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @in_3xp1_4x true pre {
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
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        %c = arith.constant 1 : i64
        %s = arith.addi %kx, %c : i64
        perfify.yield %s : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 4 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp eq, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @in_3xp1_4x true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 3X in [0, 8]: does not hold
module {
  func.func @in_0_8(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
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
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 3X lt [3X, 4X]: does not hold
module {
  func.func @lt_3x_4x(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @lt_3x_4x true pre {
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
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %ub = perfify.constant_cost {
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 4 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %iv = perfify.interval [%lb, %ub]
      %cmp = perfify.cmp lt, %cost, %iv : !perfify.cost, !perfify.interval
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @lt_3x_4x true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 3X le 3X (scalar bound): holds
module {
  func.func @le_3x_scalar(%a0: i64) -> i64 {
    %a1 = arith.muli %a0, %a0 : i64
    %a2 = arith.muli %a1, %a1 : i64
    %a3 = arith.muli %a2, %a2 : i64
    return %a3 : i64
  }
  perfify.define_symbol @X : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @le_3x_scalar true pre {
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
        %x = perfify.sym_cost @X : i64
        %k = arith.constant 3 : i64
        %kx = arith.muli %k, %x : i64
        perfify.yield %kx : i64
      } : !perfify.cost
      %cmp = perfify.cmp le, %cost, %bound
      perfify.assume %cmp
    }
  }
}

// CHECK-LABEL: perfify.conditions @le_3x_scalar true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
