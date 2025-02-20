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
// CHECK:             scf.for %[[VAL_3:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]] {
// CHECK:               "before.keepalive"(%[[VAL_3]]) : (index) -> ()
// CHECK:             }
// CHECK:             return %[[VAL_0]] : index
// CHECK:           }
// CHECK:         }

// ----

//Case: Multiple iter_args
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
// CHECK:             %[[VAL_3:.*]] = scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]] iter_args(%[[VAL_5:.*]] = %[[VAL_2]]) -> (i32)  : i32 {
// CHECK:               "before.keepalive"(%[[VAL_4]], %[[VAL_5]]) : (i32, i32) -> ()
// CHECK:               scf.yield %[[VAL_5]] : i32
// CHECK:             }
// CHECK:             return %[[VAL_3]] : i32
// CHECK:           }
// CHECK:         }

// ----

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
// CHECK-DAG:             %[[VAL_4:.*]] = arith.constant -5 : i32
// CHECK:             scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_3]]  : i32 {
// CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_0]] : i32
// CHECK:               "before.keepalive"(%[[VAL_7]]) : (i32) -> ()
// CHECK:             }
// CHECK:             return %[[VAL_4]] : i32
// CHECK:           }
// CHECK:         }

// ----

//Executes only once
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
// CHECK:             "before.keepalive"(%[[VAL_1]]) : (index) -> ()
// CHECK:             return %[[VAL_0]] : index
// CHECK:           }
// CHECK:         }

// ----

//Case: Multiple iter_args, index is second arg
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
// CHECK:             %[[VAL_3:.*]] = scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]] iter_args(%[[VAL_5:.*]] = %[[VAL_2]]) -> (i32)  : i32 {
// CHECK:               %[[VAL_6:.*]] = "before.keepalive"(%[[VAL_5]], %[[VAL_4]]) : (i32, i32) -> i32
// CHECK:               scf.yield %[[VAL_6]] : i32
// CHECK:             }
// CHECK:             return %[[VAL_0]] : i32
// CHECK:           }
// CHECK:         }

// ----

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
// CHECK:             scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:               "before.keepalive"(%[[VAL_3]]) : (index) -> ()
// CHECK:             }
// CHECK:             return %[[VAL_2]] : index
// CHECK:           }
// CHECK:         }

// ----

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
// CHECK-DAG:             %[[VAL_0:.*]] = arith.constant -5 : i32
// CHECK-DAG:             %[[VAL_1:.*]] = arith.constant 4 : i32
// CHECK-DAG:             %[[VAL_2:.*]] = arith.constant -4 : i32
// CHECK-DAG:             %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:             %[[VAL_4:.*]] = arith.constant -1 : i32
// CHECK:             scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_3]]  : i32 {
// CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_4]] : i32
// CHECK:               "before.keepalive"(%[[VAL_7]]) : (i32) -> ()
// CHECK:             }
// CHECK:             return %[[VAL_0]] : i32
// CHECK:           }
// CHECK:         }