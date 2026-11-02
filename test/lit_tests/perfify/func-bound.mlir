// RUN: enzymexlamlir-opt %s -simple-cycle-analysis | FileCheck %s

// A call to a function with a perfify.func_bound costs the given cost, whatever
// the function's body costs and with no perfify.conditions on the callee.
module {
  // really costs 2, but is given 10
  func.func @given(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  func.func private @given_external(i64) -> i64
  func.func private @unknown(i64) -> i64

  func.func @two_calls(%a0: i64) -> i64 {
    %x = func.call @given(%a0) : (i64) -> i64
    %y = func.call @given(%x) : (i64) -> i64
    return %y : i64
  }
  func.func @too_low(%a0: i64) -> i64 {
    %x = func.call @given(%a0) : (i64) -> i64
    %y = func.call @given(%x) : (i64) -> i64
    return %y : i64
  }
  func.func @external(%a0: i64) -> i64 {
    %x = func.call @given_external(%a0) : (i64) -> i64
    return %x : i64
  }
  func.func @in_loop(%a0: i64) -> i64 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %i = %lb to %ub step %step {
      %x = func.call @given(%a0) : (i64) -> i64
    }
    return %a0 : i64
  }
  func.func @no_bound(%a0: i64) -> i64 {
    %x = func.call @unknown(%a0) : (i64) -> i64
    return %x : i64
  }

  perfify.func_bound @given {
    %c = arith.constant 10 : i64
    perfify.yield %c : i64
  }
  perfify.func_bound @given_external {
    %c = arith.constant 7 : i64
    perfify.yield %c : i64
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

    // 10 + 10 == 20
    perfify.conditions @two_calls true pre {
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
        %c = arith.constant 20 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }

    // 20 <= 19 doesn't hold
    perfify.conditions @too_low true pre {
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
        %c = arith.constant 19 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }

    // a function with no body can be given a cost
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
        %c = arith.constant 7 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }

    // 3 iterations of 10
    perfify.conditions @in_loop true pre {
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
        %c = arith.constant 30 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }

    // no given cost and no proven bound, so the call stays unproven
    perfify.conditions @no_bound true pre {
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

// CHECK: perfify.conditions @two_calls
// CHECK: } post {
// CHECK: perfify.assume %{{.*}} {satres = true}
// CHECK: perfify.conditions @too_low
// CHECK: } post {
// CHECK: perfify.assume %{{.*}} {satres = false}
// CHECK: perfify.conditions @external
// CHECK: } post {
// CHECK: perfify.assume %{{.*}} {satres = true}
// CHECK: perfify.conditions @in_loop
// CHECK: } post {
// CHECK: perfify.assume %{{.*}} {satres = true}
// CHECK: perfify.conditions @no_bound
// CHECK: } post {
// CHECK: perfify.assume %{{.*}} {satres = false}
