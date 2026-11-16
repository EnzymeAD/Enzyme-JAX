// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=false})" | FileCheck %s

// A launch dimension packed into both halves of an i64 (x * 0x100000001) and
// read back: the low half through a trunc, the high half through a shift.
func.func @halves(%q: i32) -> (i32, i32) {
  %c = arith.constant 4294967297 : i64
  %c32 = arith.constant 32 : i64
  %e = arith.extui %q : i32 to i64
  %m = arith.muli %e, %c overflow<nuw> : i64
  %lo = arith.trunci %m : i64 to i32
  %hi64 = arith.shrui %m, %c32 : i64
  %hi = arith.trunci %hi64 : i64 to i32
  return %lo, %hi : i32, i32
}

// The low half of a product is only the value when the constant is 1 modulo
// 2^32; 0x100000002 doubles it.
func.func @low_bits_not_one(%q: i32) -> i32 {
  %c = arith.constant 4294967298 : i64
  %e = arith.extui %q : i32 to i64
  %m = arith.muli %e, %c : i64
  %lo = arith.trunci %m : i64 to i32
  return %lo : i32
}

// A value wider than the shift overlaps its own shifted copy.
func.func @wide_input(%q: i40) -> i64 {
  %c = arith.constant 4294967297 : i64
  %c32 = arith.constant 32 : i64
  %e = arith.extui %q : i40 to i64
  %m = arith.muli %e, %c : i64
  %hi = arith.shrui %m, %c32 : i64
  return %hi : i64
}

// CHECK:    func.func @halves(%arg0: i32) -> (i32, i32) {
// CHECK-NEXT:    return %arg0, %arg0 : i32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @low_bits_not_one(%arg0: i32) -> i32 {
// CHECK-NEXT:    %c4294967298_i64 = arith.constant 4294967298 : i64
// CHECK-NEXT:    %0 = arith.extui %arg0 : i32 to i64
// CHECK-NEXT:    %1 = arith.muli %0, %c4294967298_i64 : i64
// CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @wide_input(%arg0: i40) -> i64 {
// CHECK-NEXT:    %c4294967297_i64 = arith.constant 4294967297 : i64
// CHECK-NEXT:    %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:    %0 = arith.extui %arg0 : i40 to i64
// CHECK-NEXT:    %1 = arith.muli %0, %c4294967297_i64 : i64
// CHECK-NEXT:    %2 = arith.shrui %1, %c32_i64 : i64
// CHECK-NEXT:    return %2 : i64
// CHECK-NEXT:  }
