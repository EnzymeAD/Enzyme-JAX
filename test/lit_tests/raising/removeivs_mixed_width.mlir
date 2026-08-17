// RUN: enzymexlamlir-opt --llvm-to-affine-access --split-input-file %s | FileCheck %s

// An iteration argument stepping by a loop-invariant amount is rewritten as a
// closed form over the iteration count. The count comes out in the induction
// variable's type and the step is in the argument's, and the two need not be
// the same type -- here an i64 loop carrying an i32 sum.

func.func @narrower_arg(%ub: i64, %s: i32) -> i32 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0_i64 to %ub step %c1_i64 iter_args(%acc = %c0_i32) -> (i32) : i64 {
    %n = arith.addi %acc, %s : i32
    scf.yield %n : i32
  }
  return %r : i32
}

// CHECK-LABEL: func.func @narrower_arg
// CHECK-NEXT:    %[[T:.+]] = arith.trunci %arg0 : i64 to i32
// CHECK-NEXT:    %[[M:.+]] = arith.muli %[[T]], %arg1 : i32
// CHECK-NEXT:    return %[[M]]

// -----

// The other way round: an i32 loop carrying an i64 sum. A count is not negative
// and fit in the type it was counted in, so widening it keeps it whole.

func.func @wider_arg(%ub: i32, %s: i64) -> i64 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %r = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %c0_i64) -> (i64) : i32 {
    %n = arith.addi %acc, %s : i64
    scf.yield %n : i64
  }
  return %r : i64
}

// CHECK-LABEL: func.func @wider_arg
// CHECK-NEXT:    %[[E:.+]] = arith.extsi %arg0 : i32 to i64
// CHECK-NEXT:    %[[M:.+]] = arith.muli %[[E]], %arg1 : i64
// CHECK-NEXT:    return %[[M]]
