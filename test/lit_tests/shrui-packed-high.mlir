// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(func.func(canonicalize-loops))" %s | FileCheck %s

// The import spells a dim3(x, 1) launch configuration as
// ori(extui(x : i32 to i64), 1 << 32) and reads the second field back as
// shrui(packed, 32). Every bit the shift keeps comes from the constant, so
// the read is that constant; leaving it opaque makes the launch bound look
// like a runtime value and kernels lose their static block shape.

func.func @high(%x: i32) -> i64 {
  %c32 = arith.constant 32 : i64
  %chigh = arith.constant 4294967296 : i64
  %e = arith.extui %x : i32 to i64
  %o = arith.ori %e, %chigh : i64
  %s = arith.shrui %o, %c32 : i64
  return %s : i64
}

// CHECK-LABEL: func.func @high(
// CHECK: %[[C:.+]] = arith.constant 1 : i64
// CHECK: return %[[C]]

// The low half keeps its runtime value: a shift below the source width must
// not fold.
func.func @low(%x: i32) -> i64 {
  %c16 = arith.constant 16 : i64
  %chigh = arith.constant 4294967296 : i64
  %e = arith.extui %x : i32 to i64
  %o = arith.ori %e, %chigh : i64
  %s = arith.shrui %o, %c16 : i64
  return %s : i64
}

// CHECK-LABEL: func.func @low(
// CHECK: arith.shrui
