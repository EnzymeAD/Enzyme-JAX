// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// (a * d + b) urem d folds to b urem d only when neither operation wraps
// unsigned. libstdc++'s vector copy sizes the tail memcpy as
// (24n - 24) urem 24 with the -24 spelled as an nsw add: reducing that to
// (-24) urem 24 reads the constant as 2^64-24 and yields 16, and the copy
// loses the last 16 bytes of every element block -- Catch2's GENERATE
// handed MFEM's AMGF test uninitialized parameters that way.

module {
  func.func @no_fold_nsw_add(%n: i64) -> i64 {
    %c24 = arith.constant 24 : i64
    %c-24 = arith.constant -24 : i64
    %m = arith.muli %n, %c24 overflow<nsw, nuw> : i64
    %s = arith.addi %m, %c-24 overflow<nsw> : i64
    %r = arith.remui %s, %c24 : i64
    return %r : i64
  }
}

// CHECK-LABEL: func.func @no_fold_nsw_add
// CHECK: %[[S:.+]] = arith.addi
// CHECK: %[[R:.+]] = arith.remui %[[S]], %c24
// CHECK: return %[[R]]

// -----

module {
  func.func @fold_nuw_add(%n: i64, %b: i64) -> i64 {
    %c24 = arith.constant 24 : i64
    %m = arith.muli %n, %c24 overflow<nuw> : i64
    %s = arith.addi %m, %b overflow<nuw> : i64
    %r = arith.remui %s, %c24 : i64
    return %r : i64
  }
}

// CHECK-LABEL: func.func @fold_nuw_add
// CHECK: %[[R:.+]] = arith.remui %arg1, %c24
// CHECK: return %[[R]]

// -----

// A power-of-two divisor only reads the low bits, and a * d cannot touch
// them however anything wraps -- no nuw needed.
module {
  func.func @fold_pow2_no_nuw(%n: i64, %b: i64) -> i64 {
    %c16 = arith.constant 16 : i64
    %m = arith.muli %n, %c16 : i64
    %s = arith.addi %m, %b : i64
    %r = arith.remui %s, %c16 : i64
    return %r : i64
  }
}

// CHECK-LABEL: func.func @fold_pow2_no_nuw
// CHECK: %[[R:.+]] = arith.remui %arg1, %c16
// CHECK: return %[[R]]

// -----

// An add of two provably nonnegative values cannot wrap unsigned, so with
// an nuw multiply the fold is sound without flags on the add.
module {
  func.func @fold_nonneg_add(%m: memref<?xi64>) {
    %c6 = arith.constant 6 : index
    affine.parallel (%i, %j) = (0, 0) to (20, 6) {
      %p = arith.muli %i, %c6 overflow<nuw> : index
      %s = arith.addi %p, %j : index
      %r = arith.remui %s, %c6 : index
      %v = arith.index_cast %r : index to i64
      %z = arith.constant 0 : index
      memref.store %v, %m[%z] : memref<?xi64>
    }
    return
  }
}

// The fold chains: (i*6 + j) urem 6 -> j urem 6 -> j, leaving no remui.
// CHECK-LABEL: func.func @fold_nonneg_add
// CHECK-NOT: arith.remui
// CHECK: arith.index_cast %arg{{[0-9]+}} : index to i64
