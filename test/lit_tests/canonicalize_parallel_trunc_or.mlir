// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel)" --split-input-file | FileCheck %s

// A truncation that discards every bit an or set sees through the or: the
// dim3 packing of a kernel launch writes grid.y into the high half of an i64
// and the launch reads back the low half.
func.func @pack(%n: i32) -> i32 {
  %c255 = arith.constant 255 : i32
  %c256 = arith.constant 256 : i32
  %c4294967296 = arith.constant 4294967296 : i64
  %a = arith.addi %n, %c255 overflow<nsw> : i32
  %d = arith.divsi %a, %c256 : i32
  %e = arith.extui %d : i32 to i64
  %o = arith.ori %e, %c4294967296 {isDisjoint} : i64
  %t = arith.trunci %o : i64 to i32
  return %t : i32
}

// CHECK-LABEL: func.func @pack(
// CHECK-NOT: arith.ori
// CHECK-NOT: arith.trunci
// CHECK: %[[A:.+]] = arith.addi
// CHECK: %[[D:.+]] = arith.divsi %[[A]]
// CHECK: return %[[D]] : i32

// -----

// Bits the truncation keeps stay an or.
func.func @keepbits(%x: i64) -> i32 {
  %c3 = arith.constant 3 : i64
  %o = arith.ori %x, %c3 : i64
  %t = arith.trunci %o : i64 to i32
  return %t : i32
}

// CHECK-LABEL: func.func @keepbits(
// CHECK: arith.ori
// CHECK: arith.trunci
