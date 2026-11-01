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

// CHECK-LABEL: perfify.conditions @callee true pre {
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

// an indirect call has no known callee: does not hold
module {
  llvm.func @indirect(%a0: i64, %fp: !llvm.ptr) -> i64 {
    %x = llvm.call %fp(%a0) : !llvm.ptr, (i64) -> i64
    llvm.return %a0 : i64
  }
  perfify.assumptions {
    perfify.cost "llvm.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @indirect true pre {
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

// CHECK-LABEL: perfify.conditions @indirect true pre {
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

// CHECK-LABEL: perfify.conditions @negative_bound true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// a float cost has no integer model, so arith.muli is not free: does not hold
module {
  func.func @float_cost(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2.5 : f64
      perfify.yield %c : f64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @float_cost true pre {
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

// CHECK-LABEL: perfify.conditions @float_cost true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// arith.maxsi is unsupported, so the cost is not its last operand: does not hold
module {
  func.func @maxsi_cost(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = arith.constant 1000 : i64
      %y = arith.constant 1 : i64
      %c = arith.maxsi %x, %y : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @maxsi_cost true pre {
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
  }
}

// CHECK-LABEL: perfify.conditions @maxsi_cost true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// arith.subi is unsupported, so the cost is not its last operand: does not hold
module {
  perfify.define_symbol @X : i64
  func.func @subi_cost(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %x = perfify.sym_cost @X : i64
      %y = arith.constant 1 : i64
      %c = arith.subi %x, %y : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @subi_cost true pre {
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
  }
}

// CHECK-LABEL: perfify.conditions @subi_cost true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// a yielded block argument has no value: does not hold
module {
  func.func @yield_arg(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
    ^bb0(%x: i64):
      perfify.yield %x : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @yield_arg true pre {
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

// CHECK-LABEL: perfify.conditions @yield_arg true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an empty cost region has no value: does not hold
module {
  func.func @empty_cost(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @empty_cost true pre {
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

// CHECK-LABEL: perfify.conditions @empty_cost true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an unsupported post bound cannot be checked: does not hold
module {
  perfify.define_symbol @X : i64
  func.func @post_bound(%a0: i64) -> i64 {
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
    perfify.conditions @post_bound true pre {
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
        %x = perfify.sym_cost @X : i64
        %y = arith.constant 1 : i64
        %c = arith.subi %x, %y : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @post_bound true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an unsupported pre bound leaves the starting cost unknown: does not hold
module {
  perfify.define_symbol @X : i64
  func.func @pre_bound(%a0: i64) -> i64 {
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
    perfify.conditions @pre_bound true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %x = perfify.sym_cost @X : i64
        %y = arith.constant 1 : i64
        %c = arith.subi %x, %y : i64
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
      %cmp = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @pre_bound true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an unsupported loop hypothesis bound fails the function, even for a bound
// that an unconstrained trip count would meet
module {
  perfify.define_symbol @X : i64
  func.func @loop_bound(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %a0 : i64 to index
    %m = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %t = perfify.arg 0
      perfify.estimated_cost @trip_ub {
        %x = perfify.sym_cost @X : i64
        %y = arith.constant 1 : i64
        %c = arith.subi %x, %y : i64
        perfify.yield %c : i64
      }
      %tcmp = perfify.cmp eq, %t, @trip_ub
      perfify.hypothesis %tcmp
      %s = arith.muli %acc, %acc : i64
      scf.yield %s : i64
    }
    return %m : i64
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
    perfify.conditions @loop_bound true pre {
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
      %cmp = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @loop_bound true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an unsatisfiable precondition proves nothing, so @vacuous_callee has no bound
// for @vacuous_caller to use: neither holds
module {
  func.func @vacuous_callee(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  func.func @vacuous_caller(%a0: i64) -> i64 {
    %x = func.call @vacuous_callee(%a0) : (i64) -> i64
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
    perfify.conditions @vacuous_callee true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp lt, %b0, @pre_cost
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
    perfify.conditions @vacuous_caller true pre {
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

// CHECK-LABEL: perfify.conditions @vacuous_callee true pre {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK-LABEL: perfify.conditions @vacuous_caller true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// le reads the lower bound of an interval, so against [, 10] it is false. The
// trip count hypothesis is unsatisfiable, not a bound of 10: does not hold
module {
  func.func @loop_le_upper_only(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %a0 : i64 to index
    %m = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %t = perfify.arg 0
      perfify.estimated_cost @trip_ub {
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      perfify.interval @trip_range [max_cost: @trip_ub]
      %tcmp = perfify.cmp le, %t, @trip_range
      perfify.hypothesis %tcmp
      %s = arith.muli %acc, %acc : i64
      scf.yield %s : i64
    }
    return %m : i64
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
    perfify.conditions @loop_le_upper_only true pre {
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

// CHECK-LABEL: perfify.conditions @loop_le_upper_only true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// an empty trip count interval [20, 10] is unsatisfiable: does not hold
module {
  func.func @loop_empty_interval(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %a0 : i64 to index
    %m = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %t = perfify.arg 0
      perfify.estimated_cost @trip_ub {
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      perfify.estimated_cost @trip_lb {
        %c = arith.constant 20 : i64
        perfify.yield %c : i64
      }
      perfify.interval @trip_range [min_cost: @trip_lb, max_cost: @trip_ub]
      %tcmp = perfify.cmp eq, %t, @trip_range
      perfify.hypothesis %tcmp
      %s = arith.muli %acc, %acc : i64
      scf.yield %s : i64
    }
    return %m : i64
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
    perfify.conditions @loop_empty_interval true pre {
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

// CHECK-LABEL: perfify.conditions @loop_empty_interval true pre {
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

// an external declaration has no body to bound: does not hold
module {
  func.func private @external(i64) -> i64
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @external true pre {
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

// CHECK-LABEL: perfify.conditions @external true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// the function follows perfify.assumptions and still costs 2: does not hold
module {
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @after_assumptions true pre {
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
  }
  func.func @after_assumptions(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
}

// CHECK-LABEL: perfify.conditions @after_assumptions true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// a triple that is not verified gives its callers no bound: does not hold
module {
  func.func @unverified(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  func.func @uses_unverified(%a0: i64) -> i64 {
    %x = func.call @unverified(%a0) : (i64) -> i64
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
    perfify.conditions @unverified false pre {
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
    perfify.conditions @uses_unverified true pre {
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

// CHECK-LABEL: perfify.conditions @uses_unverified true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// @starts_at_100's post holds only from its pre, which a call does not meet, so
// @uses_start has no cost for the call: does not hold
module {
  func.func @starts_at_100(%a0: i64) -> i64 {
    return %a0 : i64
  }
  func.func @uses_start(%a0: i64) -> i64 {
    %x = func.call @starts_at_100(%a0) : (i64) -> i64
    return %x : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @starts_at_100 true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 100 : i64
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
      %cmp = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @uses_start true pre {
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
      %cmp = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @starts_at_100 true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @uses_start true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// checking @free_callee on demand must not forget that @keeps_unproven has an
// op with no cost entry: does not hold
module {
  func.func @keeps_unproven(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    %x = func.call @free_callee(%m) : (i64) -> i64
    return %x : i64
  }
  func.func @free_callee(%a0: i64) -> i64 {
    return %a0 : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @keeps_unproven true pre {
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
    perfify.conditions @free_callee true pre {
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

// CHECK-LABEL: perfify.conditions @keeps_unproven true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK-LABEL: perfify.conditions @free_callee true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// checking @leaf on demand must not hand @leaf's pre to @mid, whose pre does not
// admit a call: @outer has no cost for @mid, so it does not hold
module {
  func.func @outer(%a0: i64) -> i64 {
    %x = func.call @mid(%a0) : (i64) -> i64
    return %x : i64
  }
  func.func @mid(%a0: i64) -> i64 {
    %x = func.call @leaf(%a0) : (i64) -> i64
    return %x : i64
  }
  func.func @leaf(%a0: i64) -> i64 {
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
    perfify.conditions @outer true pre {
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
        %c = arith.constant 109 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @mid true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 100 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 109 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @leaf true pre {
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
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @outer true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK-LABEL: perfify.conditions @mid true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @leaf true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}

// -----

// mutually recursive calls have no proven bound: does not hold
module {
  func.func @ping(%a0: i64) -> i64 {
    %x = func.call @pong(%a0) : (i64) -> i64
    return %x : i64
  }
  func.func @pong(%a0: i64) -> i64 {
    %x = func.call @ping(%a0) : (i64) -> i64
    return %x : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @ping true pre {
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
    perfify.conditions @pong true pre {
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

// CHECK-LABEL: perfify.conditions @ping true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
// CHECK-LABEL: perfify.conditions @pong true pre {
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

// -----

// a second repeated symbol must not clear the first, 10 is reachable: does not hold
module {
  func.func @ne_two_inner(%a0: i64) -> i64 {
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
      %y = scf.for %i = %c0 to %u step %c1 iter_args(%w = %x) -> (i64) {
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
    perfify.conditions @ne_two_inner true pre {
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
      %cmp = perfify.cmp ne, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @ne_two_inner true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}

// -----

// one call_cost stands for both iterations, and 1 + 2 is allowed: does not hold
module {
  func.func @bounded_callee(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  func.func @ne_shared_call(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %u = arith.index_cast %a0 : i64 to index
    %r = scf.for %o = %c0 to %c2 step %c1 iter_args(%v = %a0) -> (i64) {
      %x = func.call @bounded_callee(%v) : (i64) -> i64
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
    perfify.conditions @bounded_callee true pre {
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
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
    perfify.conditions @ne_shared_call true pre {
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
        %c = arith.constant 3 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ne, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// CHECK-LABEL: perfify.conditions @bounded_callee true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK-LABEL: perfify.conditions @ne_shared_call true pre {
// CHECK:   perfify.assume %{{.*}} {satres = true}
// CHECK:   } post {
// CHECK:   perfify.assume %{{.*}} {satres = false}
