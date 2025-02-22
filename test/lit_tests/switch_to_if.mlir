// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize-loops))" | FileCheck %s

// CHECK-LABEL: func @switch_to_if
// CHECK-SAME: %[[ARG0:.+]]: index
func.func @switch_to_if(%arg0: index) -> i32 {
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32
  // CHECK-DAG: %[[C24:.+]] = arith.constant 24 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[CMPI:.+]] = arith.cmpi eq, %[[ARG0]], %[[C0]] : index
  // CHECK: %{{.+}} = arith.select %[[CMPI]], %[[C42]], %[[C24]] : i32
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

// CHECK-LABEL: func @switch_to_if2
// CHECK-SAME: %[[ARG0:.+]]: index
func.func @switch_to_if2(%arg0: index) -> i32 {
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32
  // CHECK-DAG: %[[C24:.+]] = arith.constant 24 : i32
  // CHECK-DAG: %[[C20:.+]] = arith.constant 20 : index
  // CHECK: %[[CMPI:.+]] = arith.cmpi eq, %[[ARG0]], %[[C20]] : index
  // CHECK: %{{.+}} = arith.select %[[CMPI]], %[[C42]], %[[C24]] : i32
  %0 = scf.index_switch %arg0 -> i32
    case 20 {
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
