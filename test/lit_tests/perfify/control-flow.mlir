// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s
// Cost regions must stay self-contained under CSE/canonicalization (Reactant runs both before analysis).
// RUN: enzymexlamlir-opt %s -split-input-file -cse -canonicalize -simple-cycle-analysis | FileCheck %s

// Branches, CFGs, loops other than scf.for and scf.while, and region ops that
// are not modeled. arith.muli and stablehlo.multiply cost 2, stablehlo.compare
// costs 1, and everything else is free unless a case says otherwise.

// the paths cost 2 and 4, so 2 is the most a lower bound can claim: holds
module {
  func.func @if_ge_2(%a0: i64, %k: i1) -> i64 {
    %r = scf.if %k -> (i64) {
      %x = arith.muli %a0, %a0 : i64
      scf.yield %x : i64
    } else {
      %x = arith.muli %a0, %a0 : i64
      %y = arith.muli %x, %x : i64
      scf.yield %y : i64
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
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.cond_br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.if" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @if_ge_2 true pre {
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
      %cmp = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @if_ge_2 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// the paths cost 2 and 4, so 4 is the least an upper bound can claim: holds
module {
  func.func @if_le_4(%a0: i64, %k: i1) -> i64 {
    %r = scf.if %k -> (i64) {
      %x = arith.muli %a0, %a0 : i64
      scf.yield %x : i64
    } else {
      %x = arith.muli %a0, %a0 : i64
      %y = arith.muli %x, %x : i64
      scf.yield %y : i64
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
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.cond_br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.if" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @if_le_4 true pre {
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
        %c = arith.constant 4 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @if_le_4 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// each arm runs at most once, so the sum of the arms bounds the cost: holds. This
// overapproximates: the most expensive path costs only 4
module {
  func.func @diamond_le_6(%a0: i64, %k: i1) -> i64 {
    cf.cond_br %k, ^bb1, ^bb2
  ^bb1:
    %x = arith.muli %a0, %a0 : i64
    cf.br ^bb3(%x : i64)
  ^bb2:
    %y = arith.muli %a0, %a0 : i64
    %z = arith.muli %y, %y : i64
    cf.br ^bb3(%z : i64)
  ^bb3(%r: i64):
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
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.cond_br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.if" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @diamond_le_6 true pre {
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
        %c = arith.constant 6 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @diamond_le_6 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// a block in a cycle may run any number of times: does not hold
module {
  func.func @cf_loop_le_1000(%a0: i64, %k: i1) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    cf.br ^bb1(%m : i64)
  ^bb1(%v: i64):
    cf.cond_br %k, ^bb2, ^bb3
  ^bb2:
    %s = arith.muli %v, %v : i64
    cf.br ^bb1(%s : i64)
  ^bb3:
    return %v : i64
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
    perfify.cost "arith.cmpi" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "cf.cond_br" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.if" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "scf.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.for" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @cf_loop_le_1000 true pre {
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
        %c = arith.constant 1000 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @cf_loop_le_1000 true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// at most 10 iterations * 2 + 11 condition checks * 1 = 31: holds
module {
  func.func @hlo_while_le_31(%a0: tensor<i64>) -> tensor<i64> {
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
    perfify.cost "stablehlo.constant" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "stablehlo.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "stablehlo.while" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @hlo_while_le_31 true pre {
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
        %c = arith.constant 31 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @hlo_while_le_31 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// the arms of an affine.if cost 4 and 2, and only one of them runs: holds
module {
  func.func @affine_if_le_4(%a0: i64, %i: index) -> i64 {
    %r = affine.if affine_set<(d0) : (d0 >= 0)>(%i) -> i64 {
      %x = arith.muli %a0, %a0 : i64
      %y = arith.muli %x, %x : i64
      affine.yield %y : i64
    } else {
      %x = arith.muli %a0, %a0 : i64
      affine.yield %x : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.if" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "affine.yield" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @affine_if_le_4 true pre {
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
        %c = arith.constant 4 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @affine_if_le_4 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
