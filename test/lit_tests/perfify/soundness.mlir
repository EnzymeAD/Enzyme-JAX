// RUN: enzymexlamlir-opt %s -split-input-file -simple-cycle-analysis | FileCheck %s
// Cost regions must stay self-contained under CSE/canonicalization (Reactant runs both before analysis).
// RUN: enzymexlamlir-opt %s -split-input-file -cse -canonicalize -simple-cycle-analysis | FileCheck %s

// Every case here has a bound that is false or cannot be established. A part of
// the cost that the analysis cannot model must fail the check, never count as 0.

// arith.muli has no cost entry, so @unknown_op's cost is incomplete: does not
// hold even though the bound is trivially met by the modeled ops
module {
  func.func @unknown_op(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @unknown_op true pre {
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
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @unknown_op true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// @callee's bound fails, so @caller has no cost for the call: does not hold
module {
  func.func @callee(%a0: i64) -> i64 {
    %m1 = arith.muli %a0, %a0 : i64
    %m2 = arith.muli %m1, %m1 : i64
    %m = arith.muli %m2, %m2 : i64
    return %m : i64
  }
  func.func @caller(%a0: i64) -> i64 {
    %x = func.call @callee(%a0) : (i64) -> i64
    return %x : i64
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
    perfify.conditions @callee true pre {
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
        %c = arith.constant 1 : i64
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
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @callee true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK-LABEL: perfify.conditions @caller true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// a recursive call has no proven bound: does not hold
module {
  func.func @recursive(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    %x = func.call @recursive(%m) : (i64) -> i64
    return %x : i64
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
    perfify.conditions @recursive true pre {
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
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @recursive true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// 2 le -1: a negative bound is not reinterpreted as unsigned, does not hold
module {
  func.func @negative_bound(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
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
    perfify.conditions @negative_bound true pre {
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
        %c = arith.constant -1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @negative_bound true {{.*}}pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// a negative op cost is unsatisfiable, so it cannot lower the total: does not hold
module {
  func.func @negative_cost(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant -1 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @negative_cost true pre {
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
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @negative_cost true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// the outer iterations can run the inner loop 1 and then 2 times, so the cost can
// be 6 although 4t never is: does not hold
module {
  func.func @ne_shared_trip(%a0: i64) -> i64 {
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
    perfify.conditions @ne_shared_trip true pre {
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
      %cmp = perfify.cmp ne, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @ne_shared_trip true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
