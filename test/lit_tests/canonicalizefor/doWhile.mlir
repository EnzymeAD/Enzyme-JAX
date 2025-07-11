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
// CHECK-LABEL:   module @simple {
// CHECK:           func.func @do_while() -> index {
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant 5 : index
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:             %[[ub:.+]] = ub.poison : index
// CHECK:             %[[f:.+]] = scf.for %[[VAL_3:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]] iter_args(%arg1 = %[[ub]]) -> (index) {
// CHECK:               "before.keepalive"(%[[VAL_3]]) : (index) -> ()
// CHECK:               %[[a1:.+]] = arith.addi %[[VAL_3]], %[[VAL_2]] : index
// CHECK:               scf.yield %[[a1]] : index 
// CHECK:             }
// CHECK:             return %[[f]] : index
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL:   module @multiple_iter_args {
// CHECK:           func.func @do_while() -> i32 {
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]] : i32 {
// CHECK:               "before.keepalive"(%[[VAL_4]], %[[VAL_2]]) : (i32, i32) -> ()
// CHECK:             }
// CHECK:             return %[[VAL_2]] : i32
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL:   module @negative_step {
// CHECK:           func.func @do_while() -> i32 {
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant -1 : i32
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 4 : i32
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant -4 : i32
// CHECK-DAG:             %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:             %[[ub:.+]] = ub.poison : i32
// CHECK:             %[[fr:.+]] = scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_3]] iter_args(%arg1 = %[[ub]]) -> (i32) : i32 {
// CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_0]] : i32
// CHECK:               "before.keepalive"(%[[VAL_7]]) : (i32) -> ()
// CHECK:             }
// CHECK:             return %[[fr]] : i32
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL:   module @execute_once {
// CHECK:           func.func @do_while() -> index {
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_2:.*]] = ub.poison : index
// CHECK:             %[[FOR:.*]] = scf.for %[[VAL_3:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_0]] iter_args(%{{.*}} = %[[VAL_2]]) -> (index) {
// CHECK:               "before.keepalive"(%[[VAL_3]]) : (index) -> ()
// CHECK:               %[[RES:.*]] = arith.addi %[[VAL_3]], %[[VAL_0]] : index
// CHECK:               scf.yield %[[RES]] : index
// CHECK:             }
// CHECK:             return %[[FOR]] : index
// CHECK:           }
// CHECK:         }

// ----

// Multiple iter_args, index is second arg
 module @multiple_iter_args_2{
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

// CHECK-LABEL:   module @multiple_iter_args_2 {
// CHECK:           func.func @do_while() -> i32 {
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:             %0 = ub.poison : i32
// CHECK:             %[[VAL_3:.*]]:2 = scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]] iter_args(%[[VAL_5:.*]] = %[[VAL_2]], %arg2 = %0) -> (i32, i32)  : i32 {
// CHECK:               %[[VAL_6:.*]] = "before.keepalive"(%[[VAL_5]], %[[VAL_4]]) : (i32, i32) -> i32
// CHECK:               %3 = arith.addi %arg0, %c1_i32 : i32
// CHECK:               scf.yield %[[VAL_6]], %3 : i32
// CHECK:             }
// CHECK:             return %[[VAL_3]]#1 : i32
// CHECK:           }
// CHECK:         }

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
// CHECK-LABEL:   module @cmpi_ne {
// CHECK:           func.func @do_while() -> index {
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant 5 : index
// CHECK:             %[[ub:.+]] = ub.poison : index
// CHECK:             %[[f3:.+]] = scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_1]] iter_args(%arg1 = %0) -> (index) {
// CHECK:               "before.keepalive"(%[[VAL_3]]) : (index) -> ()
// CHECK:                %[[a2:.+]] = arith.addi %[[VAL_3]], %[[VAL_1]] : index
// CHECK:                scf.yield %[[a2]] : index
// CHECK:             }
// CHECK:             return %[[f3]] : index
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL:   module @cmpi_ne_neg {
// CHECK:           func.func @do_while2() -> i32 {
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 4 : i32
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant -4 : i32
// CHECK-DAG:             %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:             %[[VAL_4:.*]] = arith.constant -1 : i32
// CHECK:             %[[ub:.+]] = ub.poison : i32
// CHECK:             %[[f4:.+]] = scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_3]] iter_args(%arg1 = %[[ub]]) -> (i32) : i32
// CHECK-NEXT:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_1]] : i32
// CHECK-NEXT:               %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_4]] : i32
// CHECK-NEXT:               "before.keepalive"(%[[VAL_7]]) : (i32) -> ()
// CHECK-NEXT:               %[[i4:.+]] = arith.addi %[[VAL_7]], %[[VAL_4]] : i32
// CHECK-NEXT:               scf.yield %[[i4]] : i32
// CHECK-NEXT:             }
// CHECK:             return %[[f4]] : i32
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL:   module @cmpi_ne_neg_step {
// CHECK:           func.func @do_while2() -> i32 {
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 49 : i32
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant -1 : i32
// CHECK-DAG:             %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:             %[[VAL_4:.*]] = arith.constant 50 : i32
// CHECK:             %[[ub:.+]] = ub.poison : i32
// CHECK:             %[[f4:.+]] = scf.for %[[VAL_5:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%arg1 = %[[ub]]) -> (i32) : i32
// CHECK-NEXT:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : i32
// CHECK-NEXT:               %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] : i32
// CHECK-NEXT:               %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_1]] : i32
// CHECK-NEXT:               "before.keepalive"(%[[VAL_8]]) : (i32) -> ()
// CHECK-NEXT:               %[[i4:.+]] = arith.addi %[[VAL_8]], %[[VAL_2]] : i32
// CHECK-NEXT:               scf.yield %[[i4]] : i32
// CHECK-NEXT:             }
// CHECK:             return %[[f4]] : i32
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL:  module @cmpi_ne_i {
// CHECK:    func.func @do_while2() -> i32 {
// CHECK-DAG:      %[[ONE:.+]] = arith.constant 1 : i32
// CHECK-DAG:      %[[C50:.+]] = arith.constant 50 : i32
// CHECK:      %[[UB:.+]] = ub.poison : i32
// CHECK-NEXT:      %[[i4:.+]] = scf.for %[[IV:.+]] = %[[ONE]] to %[[C50]] step %[[ONE]] iter_args(%[[VAL:.+]] = %[[UB]]) -> (i32)  : i32 {
// CHECK-NEXT:        "before.keepalive"(%[[IV]]) : (i32) -> ()
// CHECK-NEXT:        %[[NEW:.+]] = arith.addi %[[IV]], %[[ONE]] : i32
// CHECK-NEXT:        scf.yield %[[NEW]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[i4:.+]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

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
// CHECK:         func.func @do_while(%[[UB:.+]]: index) -> index {
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[POISON:.+]] = ub.poison : index
// CHECK-NEXT:      %[[ADJ_UB:.+]] = arith.maxsi %[[UB]], %[[C1]] : index
// CHECK-NEXT:      %[[FOR:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[ADJ_UB]] step %[[C1]] iter_args(%{{.*}} = %[[POISON]]) -> (index) {
// CHECK-NEXT:        "before.keepalive"(%[[IV]]) : (index) -> ()
// CHECK-NEXT:        %[[NEW:.+]] = arith.addi %[[IV]], %[[C1]] : index
// CHECK-NEXT:        scf.yield %[[NEW]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FOR]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:  }

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
// CHECK:         func.func @do_while(%[[LB:.+]]: index, %[[UB:.+]]: index) -> index {
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[POISON:.+]] = ub.poison : index
// CHECK-NEXT:      %[[LBP1:.+]] = arith.addi %[[LB]], %[[C1]] : index
// CHECK-NEXT:      %[[ADJ_UB:.+]] = arith.maxsi %[[LBP1]], %[[UB]] : index
// CHECK-NEXT:      %[[FOR:.+]] = scf.for %[[IV:.+]] = %[[LB]] to %[[ADJ_UB]] step %[[C1]] iter_args(%{{.*}} = %[[POISON]]) -> (index) {
// CHECK-NEXT:        "before.keepalive"(%[[IV]]) : (index) -> ()
// CHECK-NEXT:        %[[NEW:.+]] = arith.addi %[[IV]], %[[C1]] : index
// CHECK-NEXT:        scf.yield %[[NEW]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FOR]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:  }

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
// CHECK:         func.func @do_while(%[[UB:.+]]: i32) -> i32 {
// CHECK-DAG:       %[[C10:.+]] = arith.constant 10 : i32
// CHECK-DAG:       %[[C11:.+]] = arith.constant 11 : i32
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG:       %[[CM1:.+]] = arith.constant -1 : i32
// CHECK-DAG:       %[[POISON:.+]] = ub.poison : i32
// CHECK-NEXT:      %[[UBP1:.+]] = arith.addi %[[UB]], %[[C1]] : i32
// CHECK-NEXT:      %[[UBP2:.+]] = arith.addi %[[UBP1]], %[[C1]] : i32
// CHECK-NEXT:      %[[ADJ_UB:.+]] = arith.maxsi %[[UBP2]], %[[C11]] : i32
// CHECK-NEXT:      %[[FOR:.+]] = scf.for %[[IV:.+]] = %[[UBP1]] to %[[ADJ_UB]] step %[[C1]] iter_args(%{{.*}} = %[[POISON]]) -> (i32) : i32 {
// CHECK-NEXT:        %[[VAL1:.+]] = arith.subi %[[IV]], %[[UBP1]] : i32
// CHECK-NEXT:        %[[VAL2:.+]] = arith.muli %[[VAL1]], %[[CM1]] : i32
// CHECK-NEXT:        %[[VAL3:.+]] = arith.addi %[[VAL2]], %[[C10]] : i32
// CHECK-NEXT:        "before.keepalive"(%[[VAL3]]) : (i32) -> ()
// CHECK-NEXT:        %[[NEW:.+]] = arith.addi %[[VAL3]], %[[CM1]] : i32
// CHECK-NEXT:        scf.yield %[[NEW]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FOR]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

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
// CHECK:         func.func @do_while(%[[UB:.+]]: index) -> (index, index) {
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:       %[[POISON:.+]] = ub.poison : index
// CHECK-NEXT:      %[[ADJ_UB:.+]] = arith.maxsi %[[UB]], %[[C1]] : index
// CHECK-NEXT:      %[[FOR:.+]]:2 = scf.for %[[IV:.+]] = %[[C0]] to %[[ADJ_UB]] step %[[C1]] iter_args(%{{.*}} = %[[POISON]], %{{.*}} = %[[POISON]]) -> (index, index) {
// CHECK-NEXT:        %[[SUM:.+]] = arith.muli %[[IV]], %[[C2]] : index
// CHECK-NEXT:        "before.keepalive"(%[[IV]], %[[SUM]]) : (index, index) -> ()
// CHECK-NEXT:        %[[NEW:.+]] = arith.addi %[[IV]], %[[C1]] : index
// CHECK-NEXT:        %[[NEW_SUM:.+]] = arith.addi %[[SUM]], %[[C2]] : index
// CHECK-NEXT:        scf.yield %[[NEW]], %[[NEW_SUM]] : index, index
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FOR]]#0, %[[FOR]]#1 : index, index
// CHECK-NEXT:    }
// CHECK-NEXT:  }
