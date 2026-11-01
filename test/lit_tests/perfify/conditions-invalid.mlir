// RUN: enzymexlamlir-opt %s -split-input-file -verify-diagnostics

// a second post hypothesis would not be the one that is checked
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{requires exactly one perfify.hypothesis in each region}}
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
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %cmp2 = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      %h2 = perfify.hypothesis %cmp2 post
      perfify.assume %h
    }
  }
}

// -----

// pre hypotheses are not conjoined: use one interval or a separate perfify.conditions
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{requires exactly one perfify.hypothesis in each region}}
    perfify.conditions @f true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %cmp2 = perfify.cmp le, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      %h2 = perfify.hypothesis %cmp2 pre
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

// -----

// a region with no hypothesis checks nothing
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{requires exactly one perfify.hypothesis in each region}}
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
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      perfify.assume %cmp
    }
  }
}

// -----

// the assume must check the region's hypothesis, not the bare comparison
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{each region must assume its perfify.hypothesis}}
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
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %cmp
    }
  }
}

// -----

// a post with no perfify.fn_cost never looks at the function
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{postcondition must compare its perfify.fn_cost}}
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
      %b0 = perfify.arg 0
      perfify.estimated_cost @post_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %b0, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// -----

// a post must compare the function's cost, not a perfify.arg
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{postcondition must compare its perfify.fn_cost}}
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
      %b0 = perfify.arg 0
      perfify.estimated_cost @post_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %b0, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }
  }
}

// -----

// a second perfify.fn_cost would count the function's cost twice
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{postcondition must compare its perfify.fn_cost}}
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
      %cost2 = perfify.fn_cost : !perfify.cost
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

// -----

// the starting cost is compared before the function runs
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{precondition must compare a perfify.arg}}
    perfify.conditions @f true pre {
      %b0 = perfify.arg 0
      %cost = perfify.fn_cost : !perfify.cost
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

// -----

// a precondition compares a perfify.arg, not a perfify.fn_cost
module {
  func.func @f() {
    return
  }
  perfify.assumptions {
    // expected-error @+1 {{precondition must compare a perfify.arg}}
    perfify.conditions @f true pre {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %cost, @pre_cost
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
