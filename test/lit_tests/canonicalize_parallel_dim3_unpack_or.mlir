// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=false})" | FileCheck %s

// Two launch dimensions packed as a disjoint or, (y << 32) | 2: the low half
// is the constant, the high half is y.
func.func @constant_low(%y: i32) -> (i32, i32) {
  %c2 = arith.constant 2 : i64
  %c32 = arith.constant 32 : i64
  %e = arith.extui %y : i32 to i64
  %s = arith.shli %e, %c32 overflow<nuw> : i64
  %p = arith.ori %s, %c2 {isDisjoint} : i64
  %lo = arith.trunci %p : i64 to i32
  %hi64 = arith.shrui %p, %c32 : i64
  %hi = arith.trunci %hi64 : i64 to i32
  return %lo, %hi : i32, i32
}

// The same dimension in both halves, (y << 32) | y.
func.func @replicated(%y: i32) -> (i32, i32) {
  %c32 = arith.constant 32 : i64
  %e = arith.extui %y : i32 to i64
  %s = arith.shli %e, %c32 overflow<nuw> : i64
  %p = arith.ori %s, %e {isDisjoint} : i64
  %lo = arith.trunci %p : i64 to i32
  %hi64 = arith.shrui %p, %c32 : i64
  %hi = arith.trunci %hi64 : i64 to i32
  return %lo, %hi : i32, i32
}

// A shifted-in side wider than its half overlaps the other: left alone.
func.func @wide_high(%y: i40, %x: i32) -> i64 {
  %c32 = arith.constant 32 : i64
  %e = arith.extui %y : i40 to i64
  %s = arith.shli %e, %c32 : i64
  %f = arith.extui %x : i32 to i64
  %p = arith.ori %s, %f : i64
  %hi = arith.shrui %p, %c32 : i64
  return %hi : i64
}

// A shift that promises no lost bits (nuw) is proof enough on its own.
func.func @nuw_high(%y: i64, %x: i32) -> i64 {
  %c32 = arith.constant 32 : i64
  %s = arith.shli %y, %c32 overflow<nuw> : i64
  %f = arith.extui %x : i32 to i64
  %p = arith.ori %s, %f : i64
  %hi = arith.shrui %p, %c32 : i64
  return %hi : i64
}

// CHECK:    func.func @constant_low(%arg0: i32) -> (i32, i32) {
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    return %c2_i32, %arg0 : i32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @replicated(%arg0: i32) -> (i32, i32) {
// CHECK-NEXT:    return %arg0, %arg0 : i32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @wide_high(%arg0: i40, %arg1: i32) -> i64 {
// CHECK-NEXT:    %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:    %0 = arith.extui %arg0 : i40 to i64
// CHECK-NEXT:    %1 = arith.shli %0, %c32_i64 : i64
// CHECK-NEXT:    %2 = arith.extui %arg1 : i32 to i64
// CHECK-NEXT:    %3 = arith.ori %1, %2 : i64
// CHECK-NEXT:    %4 = arith.shrui %3, %c32_i64 : i64
// CHECK-NEXT:    return %4 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @nuw_high(%arg0: i64, %arg1: i32) -> i64 {
// CHECK-NEXT:    return %arg0 : i64
// CHECK-NEXT:  }
