// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize-loops))" | FileCheck %s

// CHECK-LABEL: func @switch_to_if
func.func @switch_to_if(%arg0: index) -> i32 {
  // CHECK-DAG: %[[CONST:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL1:.*]] = arith.constant 42 : i32
  // CHECK-DAG: %[[VAL2:.*]] = arith.constant 24 : i32
  // CHECK: %[[CMP:.*]] = arith.cmpi eq, %arg0, %[[CONST]] : index
  // CHECK: %[[RESULT:.*]] = scf.if %[[CMP]] -> (i32) {
  // CHECK:   scf.yield %[[VAL1]] : i32
  // CHECK: } else {
  // CHECK:   scf.yield %[[VAL2]] : i32
  // CHECK: }
  // CHECK: return %[[RESULT]] : i32
  %0 = scf.index_switch %arg0 -> i32
    case 0 {
      %1 = arith.constant 42 : i32
      scf.yield %1 : i32
    }
    default {
      %1 = arith.constant 24 : i32
      scf.yield %1 : i32
    }
  return %0 : i32
}

// Should not convert switches with more than 1 case
// CHECK-LABEL: func @switch_two_cases
func.func @switch_two_cases(%arg0: index) -> i32 {
  // CHECK: scf.index_switch
  %0 = scf.index_switch %arg0 -> i32
    case 0 {
      %1 = arith.constant 42 : i32
      scf.yield %1 : i32
    }
    case 1 {
      %1 = arith.constant 24 : i32
      scf.yield %1 : i32
    }
    default {
      %1 = arith.constant 0 : i32
      scf.yield %1 : i32
    }
  return %0 : i32
} 
