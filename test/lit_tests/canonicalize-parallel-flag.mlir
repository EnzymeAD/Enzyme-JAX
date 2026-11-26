// RUN: enzymexlamlir-opt %s --canonicalize-parallel | FileCheck %s --check-prefixes=CHECK,NOCSE
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=true})" | FileCheck %s --check-prefixes=CHECK,NOCSE
// RUN: enzymexlamlir-opt %s --canonicalize-parallel="cse-between-iterations=true" | FileCheck %s --check-prefixes=CHECK,CSE
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=true cse-between-iterations=true})" | FileCheck %s --check-prefixes=CHECK,CSE

module {
  func.func @f(%arg0: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %0 = arith.addi %arg0, %c0 : i64
    return %0 : i64
  }
  func.func @g(%arg0: i64) -> i64 {
    %c1 = arith.constant 1 : i64
    %0 = arith.muli %arg0, %c1 : i64
    return %0 : i64
  }
  func.func @cse_exposes_fold(%arg0: i64, %arg1: i64) -> i64 {
    %0 = arith.addi %arg0, %arg1 : i64
    %1 = arith.addi %arg0, %arg1 : i64
    %2 = arith.subi %0, %1 : i64
    return %2 : i64
  }
}

// CHECK-LABEL: func.func @f
// CHECK-NEXT: return %arg0 : i64
// CHECK-LABEL: func.func @g
// CHECK-NEXT: return %arg0 : i64
// NOCSE-LABEL: func.func @cse_exposes_fold
// NOCSE-NEXT: %[[A0:.+]] = arith.addi
// NOCSE-NEXT: %[[A1:.+]] = arith.addi
// NOCSE-NEXT: %[[SUB:.+]] = arith.subi %[[A0]], %[[A1]] : i64
// NOCSE-NEXT: return %[[SUB]] : i64
// CSE-LABEL: func.func @cse_exposes_fold
// CSE-NEXT: %[[C0:.+]] = arith.constant 0 : i64
// CSE-NEXT: return %[[C0]] : i64
