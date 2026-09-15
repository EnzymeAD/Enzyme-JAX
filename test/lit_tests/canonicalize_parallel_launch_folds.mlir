// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=false})" | FileCheck %s

// A grid dimension packed with a constant high half, (zext(n) | (5 << 32)),
// read back by the shift: the constant.
func.func @grid_high_half(%n: i32) -> i64 {
  %c = arith.constant 21474836480 : i64
  %c32 = arith.constant 32 : i64
  %e = arith.extui %n : i32 to i64
  %p = arith.ori %e, %c : i64
  %hi = arith.shrui %p, %c32 : i64
  return %hi : i64
}

// A block dimension max(min(a, 14), min(b, 14)): the clamp moves outside.
func.func @max_of_mins(%a: i32, %b: i32) -> i32 {
  %c14 = arith.constant 14 : i32
  %x = arith.minsi %a, %c14 : i32
  %y = arith.minsi %b, %c14 : i32
  %m = arith.maxsi %x, %y : i32
  return %m : i32
}

// The dual, and the unsigned form.
func.func @min_of_maxs(%a: i32, %b: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %x = arith.maxui %a, %c1 : i32
  %y = arith.maxui %b, %c1 : i32
  %m = arith.minui %x, %y : i32
  return %m : i32
}

// Different clamps do not combine.
func.func @different_clamps(%a: i32, %b: i32) -> i32 {
  %c14 = arith.constant 14 : i32
  %c9 = arith.constant 9 : i32
  %x = arith.minsi %a, %c14 : i32
  %y = arith.minsi %b, %c9 : i32
  %m = arith.maxsi %x, %y : i32
  return %m : i32
}

// The clamp need not be a constant, only shared, in either operand position.
func.func @shared_value(%a: i32, %b: i32, %c: i32) -> i32 {
  %x = arith.minsi %c, %a : i32
  %y = arith.minsi %b, %c : i32
  %m = arith.maxsi %x, %y : i32
  return %m : i32
}

// CHECK:    func.func @grid_high_half(%arg0: i32) -> i64 {
// CHECK-NEXT:    %c5_i64 = arith.constant 5 : i64
// CHECK-NEXT:    return %c5_i64 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @max_of_mins(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:    %c14_i32 = arith.constant 14 : i32
// CHECK-NEXT:    %0 = arith.maxsi %arg0, %arg1 : i32
// CHECK-NEXT:    %1 = arith.minsi %0, %c14_i32 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @min_of_maxs(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %0 = arith.minui %arg0, %arg1 : i32
// CHECK-NEXT:    %1 = arith.maxui %0, %c1_i32 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @different_clamps(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:    %c14_i32 = arith.constant 14 : i32
// CHECK-NEXT:    %c9_i32 = arith.constant 9 : i32
// CHECK-NEXT:    %0 = arith.minsi %arg0, %c14_i32 : i32
// CHECK-NEXT:    %1 = arith.minsi %arg1, %c9_i32 : i32
// CHECK-NEXT:    %2 = arith.maxsi %0, %1 : i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @shared_value(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
// CHECK-NEXT:    %0 = arith.maxsi %arg0, %arg1 : i32
// CHECK-NEXT:    %1 = arith.minsi %0, %arg2 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
