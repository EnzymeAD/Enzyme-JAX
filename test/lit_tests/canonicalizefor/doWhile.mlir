// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for -split-input-file %s | FileCheck %s
module @simple{
  func.func @do_while() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index

    %result = scf.while (%i = %c0) : (index) -> index {
      "before.keepalive"(%i) : (index) -> ()
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %updated, %c5 : index
      scf.condition(%cond) %updated : index
    } do {
    ^bb0(%new_i: index):
      scf.yield %new_i : index
    }
    
    return %result : index
  }
}
// CHECK-LABEL: module @simple {
// CHECK-NEXT:   func.func @do_while() -> index {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c6 = arith.constant 6 : index
// CHECK-NEXT:     %c5 = arith.constant 5 : index
// CHECK-NEXT:     scf.for %arg0 = %c1 to %c6 step %c1 {
// CHECK-NEXT:       %0 = arith.subi %arg0, %c1 : index
// CHECK-NEXT:       "before.keepalive"(%0) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c5 : index
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Multiple iter_args
module @multiple_iter_args{
  func.func @do_while() -> i32 {
    %init_count = arith.constant 0 : i32
    %init_sum = arith.constant 1 : i32
    %c1 = arith.constant 1 : i32
    %final_count, %final_sum = scf.while (%count = %init_count, %sum = %init_sum) : (i32, i32) -> (i32, i32) {
      %threshold = arith.constant 10 : i32
      "before.keepalive"(%count, %sum) : (i32, i32) -> ()
      %updated = arith.addi %count, %c1 : i32
      %count_lt = arith.cmpi slt, %updated, %threshold : i32
      scf.condition(%count_lt) %updated, %sum : i32, i32
    } do {
    ^bb0( %current_count: i32, %current_sum: i32):
      scf.yield %current_count, %current_sum : i32, i32
    }
    return %final_sum : i32
  }
}

// CHECK-LABEL: module @multiple_iter_args {
// CHECK-NEXT:   func.func @do_while() -> i32 {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c11_i32 = arith.constant 11 : i32
// CHECK-NEXT:     scf.for %arg0 = %c1_i32 to %c11_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       %0 = arith.addi %arg0, %c-1_i32 : i32
// CHECK-NEXT:       "before.keepalive"(%0, %c1_i32) : (i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c1_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Negative step
module @negative_step{
  func.func @do_while() -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c-5 = arith.constant -5 : i32

    %result = scf.while (%i = %c0) : (i32) -> i32 {
      "before.keepalive"(%i) : (i32) -> ()
      %updated = arith.subi %i, %c1 : i32
      %cond = arith.cmpi sgt, %updated, %c-5 : i32
      scf.condition(%cond) %updated : i32
    } do {
    ^bb0(%new_i: i32):
      scf.yield %new_i : i32
    }
    
    return %result : i32
  }
}

// CHECK-LABEL: module @negative_step {
// CHECK-NEXT:   func.func @do_while() -> i32 {
// CHECK-NEXT:     %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c-4_i32 = arith.constant -4 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c-5_i32 = arith.constant -5 : i32
// CHECK-NEXT:     scf.for %arg0 = %c-4_i32 to %c1_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       %0 = arith.addi %arg0, %c4_i32 : i32
// CHECK-NEXT:       %1 = arith.muli %0, %c-1_i32 : i32
// CHECK-NEXT:       "before.keepalive"(%1) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c-5_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Executes only once
module @execute_once{
  func.func @do_while() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index

    %result = scf.while (%i = %c0) : (index) -> index {
      "before.keepalive"(%i) : (index) -> ()
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %updated, %c1 : index
      scf.condition(%cond) %updated : index
    } do {
    ^bb0(%new_i: index):
      scf.yield %new_i : index
    }
    
    return %result : index
  }
}

// CHECK-LABEL: module @execute_once {
// CHECK-NEXT:   func.func @do_while() -> index {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     scf.for %arg0 = %c1 to %c2 step %c1 {
// CHECK-NEXT:       %0 = arith.subi %arg0, %c1 : index
// CHECK-NEXT:       "before.keepalive"(%0) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c1 : index
// CHECK-NEXT:   }
// CHECK-NEXT: }

// ----

// Multiple iter_args, index is second arg
 module @multiple_iter_args_2 {
   func.func @do_while() -> i32 {
     %init_count = arith.constant 0 : i32
     %init_sum = arith.constant 1 : i32
     %c1 = arith.constant 1 : i32
     %final_count, %final_sum = scf.while (%sum = %init_sum, %count = %init_count) : (i32, i32) -> (i32, i32) {
       %threshold = arith.constant 10 : i32
       %sum2 = "before.keepalive"(%sum, %count) : (i32, i32) -> (i32)
       %updated = arith.addi %count, %c1 : i32
       %count_lt = arith.cmpi slt, %updated, %threshold : i32
       scf.condition(%count_lt) %sum2, %updated: i32, i32
     } do {
     ^bb0(%current_sum: i32, %current_count: i32):
       scf.yield %current_sum, %current_count : i32, i32
     }
     return %final_sum : i32
   }
 }

// CHECK-LABEL: module @multiple_iter_args_2 {
// CHECK-NEXT:   func.func @do_while() -> i32 {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c11_i32 = arith.constant 11 : i32
// CHECK-NEXT:     %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:     %0 = scf.for %arg0 = %c1_i32 to %c11_i32 step %c1_i32 iter_args(%arg1 = %c1_i32) -> (i32)  : i32 {
// CHECK-NEXT:       %1 = arith.addi %arg0, %c-1_i32 : i32
// CHECK-NEXT:       %2 = "before.keepalive"(%arg1, %1) : (i32, i32) -> i32
// CHECK-NEXT:       scf.yield %2 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c10_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

module @cmpi_ne{
  func.func @do_while() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index

    %result = scf.while (%i = %c0) : (index) -> index {
      "before.keepalive"(%i) : (index) -> ()
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi ne, %updated, %c5 : index
      scf.condition(%cond) %updated : index
    } do {
    ^bb0(%new_i: index):
      scf.yield %new_i : index
    }

    return %result : index
  }
}
// CHECK-LABEL: module @cmpi_ne {
// CHECK-NEXT:   func.func @do_while() -> index {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c6 = arith.constant 6 : index
// CHECK-NEXT:     %c5 = arith.constant 5 : index
// CHECK-NEXT:     scf.for %arg0 = %c1 to %c6 step %c1 {
// CHECK-NEXT:       %0 = arith.subi %arg0, %c1 : index
// CHECK-NEXT:       "before.keepalive"(%0) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c5 : index
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

module @cmpi_ne_neg{
  func.func @do_while2() -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c-5 = arith.constant -5 : i32

    %result = scf.while (%i = %c0) : (i32) -> i32 {
      "before.keepalive"(%i) : (i32) -> ()
      %updated = arith.subi %i, %c1 : i32
      %cond = arith.cmpi ne, %updated, %c-5 : i32
      scf.condition(%cond) %updated : i32
    } do {
    ^bb0(%new_i: i32):
      scf.yield %new_i : i32
    }

    return %result : i32
  }
}

// CHECK-LABEL: module @cmpi_ne_neg {
// CHECK-NEXT:   func.func @do_while2() -> i32 {
// CHECK-NEXT:     %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c-4_i32 = arith.constant -4 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c-5_i32 = arith.constant -5 : i32
// CHECK-NEXT:     scf.for %arg0 = %c-4_i32 to %c1_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       %0 = arith.addi %arg0, %c4_i32 : i32
// CHECK-NEXT:       %1 = arith.muli %0, %c-1_i32 : i32
// CHECK-NEXT:       "before.keepalive"(%1) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c-5_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

module @cmpi_ne_neg_step{
  func.func @do_while2() -> i32 {
    %start = arith.constant 49 : i32
    %step = arith.constant -1 : i32
    %stop = arith.constant 1 : i32
  
    %result = scf.while (%i = %start) : (i32) -> i32 {
      "before.keepalive"(%i) : (i32) -> ()
      %updated = arith.addi %i, %step : i32
      %cond = arith.cmpi ne, %i, %stop : i32
      scf.condition(%cond) %updated : i32
    } do {
    ^bb0(%new_i: i32):
      scf.yield %new_i : i32
    }
  
    return %result : i32
  }
}

// CHECK-LABEL: module @cmpi_ne_neg_step {
// CHECK-NEXT:   func.func @do_while2() -> i32 {
// CHECK-NEXT:     %c49_i32 = arith.constant 49 : i32
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c50_i32 = arith.constant 50 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     scf.for %arg0 = %c1_i32 to %c50_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       %0 = arith.addi %arg0, %c-1_i32 : i32
// CHECK-NEXT:       %1 = arith.muli %0, %c-1_i32 : i32
// CHECK-NEXT:       %2 = arith.addi %1, %c49_i32 : i32
// CHECK-NEXT:       "before.keepalive"(%2) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

module @cmpi_ne_i{
  func.func @do_while2() -> i32 {
    %start = arith.constant 1 : i32
    %step = arith.constant 1 : i32
    %stop = arith.constant 49 : i32
  
    %result = scf.while (%i = %start) : (i32) -> i32 {
      "before.keepalive"(%i) : (i32) -> ()
      %updated = arith.addi %i, %step : i32
      %cond = arith.cmpi ne, %i, %stop : i32
      scf.condition(%cond) %updated : i32
    } do {
    ^bb0(%new_i: i32):
      scf.yield %new_i : i32
    }
  
    return %result : i32
  }
}

// CHECK-LABEL: module @cmpi_ne_i {
// CHECK-NEXT:   func.func @do_while2() -> i32 {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c50_i32 = arith.constant 50 : i32
// CHECK-NEXT:     scf.for %arg0 = %c1_i32 to %c50_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       "before.keepalive"(%arg0) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c50_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Non-constant upper bound
module @test_dynamic_ub {
  func.func @do_while(%ub: index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.while (%i = %c0) : (index) -> index {
      "before.keepalive"(%i) : (index) -> ()
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %updated, %ub : index
      scf.condition(%cond) %updated : index
    } do {
    ^bb0(%new_i: index):
      scf.yield %new_i : index
    }

    return %result : index
  }
}

// CHECK-LABEL: module @test_dynamic_ub {
// CHECK-NEXT:   func.func @do_while(%arg0: index) -> index {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = arith.maxsi %arg0, %c1 : index
// CHECK-NEXT:     %1 = arith.addi %0, %c1 : index
// CHECK-NEXT:     %2 = arith.maxsi %0, %c0 : index
// CHECK-NEXT:     scf.for %arg1 = %c1 to %1 step %c1 {
// CHECK-NEXT:       %3 = arith.subi %arg1, %c1 : index
// CHECK-NEXT:       "before.keepalive"(%3) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Non-constant lower bound and upper bound
module @test_fully_dynamic {
  func.func @do_while(%lb: index, %ub: index) -> index {
    %c1 = arith.constant 1 : index
    %result = scf.while (%i = %lb) : (index) -> index {
      "before.keepalive"(%i) : (index) -> ()
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %updated, %ub : index
      scf.condition(%cond) %updated : index
    } do {
    ^bb0(%new_i: index):
      scf.yield %new_i : index
    }

    return %result : index
  }
}

// CHECK-LABEL: module @test_fully_dynamic {
// CHECK-NEXT:   func.func @do_while(%arg0: index, %arg1: index) -> index {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = arith.addi %arg0, %c1 : index
// CHECK-NEXT:     %1 = arith.maxsi %arg1, %0 : index
// CHECK-NEXT:     %2 = arith.addi %1, %c1 : index
// CHECK-NEXT:     %3 = arith.subi %2, %0 : index
// CHECK-NEXT:     %4 = arith.maxsi %3, %c0 : index
// CHECK-NEXT:     %5 = arith.subi %4, %c1 : index
// CHECK-NEXT:     %6 = arith.addi %arg0, %5 : index
// CHECK-NEXT:     %7 = arith.addi %6, %c1 : index
// CHECK-NEXT:     scf.for %arg2 = %0 to %2 step %c1 {
// CHECK-NEXT:       %8 = arith.subi %arg2, %0 : index
// CHECK-NEXT:       %9 = arith.addi %arg0, %8 : index
// CHECK-NEXT:       "before.keepalive"(%9) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %7 : index
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Negative step with dynamic upper bound
module @test_negative_step_dynamic_ub {
  func.func @do_while(%ub: i32) -> i32 {
    %c10 = arith.constant 10 : i32
    %c-1 = arith.constant -1 : i32
    %result = scf.while (%i = %c10) : (i32) -> i32 {
      "before.keepalive"(%i) : (i32) -> ()
      %updated = arith.addi %i, %c-1 : i32
      %cond = arith.cmpi sgt, %updated, %ub : i32
      scf.condition(%cond) %updated : i32
    } do {
    ^bb0(%new_i: i32):
      scf.yield %new_i : i32
    }

    return %result : i32
  }
}

// CHECK-LABEL: module @test_negative_step_dynamic_ub {
// CHECK-NEXT:   func.func @do_while(%arg0: i32) -> i32 {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = arith.addi %arg0, %c1_i32 : i32
// CHECK-NEXT:     %1 = arith.maxsi %0, %c10_i32 : i32
// CHECK-NEXT:     %2 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:     %3 = arith.subi %2, %0 : i32
// CHECK-NEXT:     %4 = arith.maxsi %3, %c0_i32 : i32
// CHECK-NEXT:     %5 = arith.addi %4, %c-1_i32 : i32
// CHECK-NEXT:     %6 = arith.muli %5, %c-1_i32 : i32
// CHECK-NEXT:     %7 = arith.addi %6, %c10_i32 : i32
// CHECK-NEXT:     %8 = arith.addi %7, %c-1_i32 : i32
// CHECK-NEXT:     scf.for %arg1 = %0 to %2 step %c1_i32  : i32 {
// CHECK-NEXT:       %9 = arith.subi %arg1, %0 : i32
// CHECK-NEXT:       %10 = arith.muli %9, %c-1_i32 : i32
// CHECK-NEXT:       %11 = arith.addi %10, %c10_i32 : i32
// CHECK-NEXT:       "before.keepalive"(%11) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %8 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Multiple iter_args with dynamic bounds
module @test_multiple_args_dynamic {
  func.func @do_while(%ub: index) -> (index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result:2 = scf.while (%i = %c0, %sum = %c0) : (index, index) -> (index, index) {
      "before.keepalive"(%i, %sum) : (index, index) -> ()
      %updated_i = arith.addi %i, %c1 : index
      %updated_sum = arith.addi %sum, %c2 : index
      %cond = arith.cmpi slt, %updated_i, %ub : index
      scf.condition(%cond) %updated_i, %updated_sum : index, index
    } do {
    ^bb0(%new_i: index, %new_sum: index):
      scf.yield %new_i, %new_sum : index, index
    }

    return %result#0, %result#1 : index, index
  }
}

// CHECK-LABEL: module @test_multiple_args_dynamic {
// CHECK-NEXT:   func.func @do_while(%arg0: index) -> (index, index) {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %0 = arith.maxsi %arg0, %c1 : index
// CHECK-NEXT:     %1 = arith.addi %0, %c1 : index
// CHECK-NEXT:     %2 = arith.maxsi %0, %c0 : index
// CHECK-NEXT:     %3 = arith.maxsi %0, %c0 : index
// CHECK-NEXT:     %4 = arith.subi %3, %c1 : index
// CHECK-NEXT:     %5 = arith.muli %4, %c2 : index
// CHECK-NEXT:     %6 = arith.addi %5, %c2 : index
// CHECK-NEXT:     scf.for %arg1 = %c1 to %1 step %c1 {
// CHECK-NEXT:       %7 = arith.subi %arg1, %c1 : index
// CHECK-NEXT:       %8 = arith.muli %7, %c2 : index
// CHECK-NEXT:       %9 = arith.subi %arg1, %c1 : index
// CHECK-NEXT:       "before.keepalive"(%9, %8) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2, %6 : index, index
// CHECK-NEXT:   }
// CHECK-NEXT: }

//----

// Loop condition is an and expression
module @test_and_condition {
  func.func @do_while(%ub : i32) -> (i32, f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %2:3 = scf.while (%arg10 = %c0_i32, %arg12 = %cst, %ac = %true) : (i32, f32, i1) -> (i32, f32, i1) {
      %3 = arith.cmpi ult, %arg10, %ub : i32
      %a = arith.andi %3, %ac : i1
      %p = arith.addi %arg10, %c1_i32 : i32
      %c = "test.something"() : () -> (i1)
      %4 = arith.addf %arg12, %cst1 : f32
      scf.condition(%a) %p, %4, %c : i32, f32, i1
    } do {
    ^bb0(%arg10: i32, %arg12: f32, %ac: i1):
      scf.yield %arg10, %arg12, %ac : i32, f32, i1
    }
    return %2#0, %2#1 : i32, f32
  }
}

// CHECK-LABEL: module @test_and_condition {
// CHECK-NEXT:   func.func @do_while(%arg0: i32) -> (i32, f32) {
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = ub.poison : i32
// CHECK-NEXT:     %1 = ub.poison : f32
// CHECK-NEXT:     %2 = ub.poison : i1
// CHECK-NEXT:     %3 = arith.maxsi %arg0, %c0_i32 : i32
// CHECK-NEXT:     %4 = arith.addi %3, %c1_i32 : i32
// CHECK-NEXT:     %5:7 = scf.for %arg1 = %c0_i32 to %4 step %c1_i32 iter_args(%arg2 = %c0_i32, %arg3 = %cst, %arg4 = %true, %arg5 = %0, %arg6 = %1, %arg7 = %2, %arg8 = %true) -> (i32, f32, i1, i32, f32, i1, i1)  : i32 {
// CHECK-NEXT:       %6:4 = scf.if %arg8 -> (i32, f32, i1, i1) {
// CHECK-NEXT:         %10 = arith.addi %arg2, %c1_i32 : i32
// CHECK-NEXT:         %11 = "test.something"() : () -> i1
// CHECK-NEXT:         %12 = arith.addf %arg3, %cst_0 : f32
// CHECK-NEXT:         scf.yield %10, %12, %11, %arg4 : i32, f32, i1, i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %arg5, %arg6, %arg7, %false : i32, f32, i1, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       %7 = arith.cmpi slt, %arg1, %arg0 : i32
// CHECK-NEXT:       %8 = arith.andi %7, %6#3 : i1
// CHECK-NEXT:       %9:3 = scf.if %8 -> (i32, f32, i1) {
// CHECK-NEXT:         scf.yield %6#0, %6#1, %6#2 : i32, f32, i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %0, %1, %2 : i32, f32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %9#0, %9#1, %9#2, %6#0, %6#1, %6#2, %6#3 : i32, f32, i1, i32, f32, i1, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return %5#3, %5#4 : i32, f32
// CHECK-NEXT:   }
// CHECK-NEXT: }
