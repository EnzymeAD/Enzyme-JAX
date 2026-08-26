// RUN: enzymexlamlir-opt %s --canonicalize-parallel | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=true})" | FileCheck %s

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
}

// CHECK-LABEL: func.func @f
// CHECK-NEXT: return %arg0 : i64
// CHECK-LABEL: func.func @g
// CHECK-NEXT: return %arg0 : i64
