// RUN: enzymexlamlir-opt %s --canonicalize-parallel --allow-unregistered-dialect | FileCheck %s

// Clang's rotation of a single-trip loop with early exits: the exit test is
// the scf.if on the carried flag, the flag is yielded false, and the else
// branch yields false, so the second evaluation exits.
func.func @flag_if(%buf: memref<8xf64>, %out: memref<i32>, %go: i1) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %v = arith.constant 2.0 : f64
  %r:2 = scf.while (%flag = %true, %res = %c0_i32) : (i1, i32) -> (i32, i1) {
    %pair:2 = scf.if %flag -> (i32, i1) {
      affine.store %v, %buf[0] : memref<8xf64>
      %again = arith.andi %go, %true : i1
      scf.yield %c5_i32, %again : i32, i1
    } else {
      scf.yield %res, %false : i32, i1
    }
    scf.condition(%pair#1) %pair#0, %pair#1 : i32, i1
  } do {
  ^bb0(%res: i32, %again: i1):
    scf.yield %false, %res : i1, i32
  }
  memref.store %r#0, %out[] : memref<i32>
  return
}

// The flag is yielded true, and the then branch's continue value is not
// constant: the loop may go on.
func.func @flag_true(%buf: memref<8xf64>, %out: memref<i32>, %go: i1) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %v = arith.constant 2.0 : f64
  %r:2 = scf.while (%flag = %true, %res = %c0_i32) : (i1, i32) -> (i32, i1) {
    %pair:2 = scf.if %flag -> (i32, i1) {
      affine.store %v, %buf[0] : memref<8xf64>
      %again = arith.andi %go, %true : i1
      scf.yield %c5_i32, %again : i32, i1
    } else {
      scf.yield %res, %false : i32, i1
    }
    scf.condition(%pair#1) %pair#0, %pair#1 : i32, i1
  } do {
  ^bb0(%res: i32, %again: i1):
    scf.yield %true, %res : i1, i32
  }
  memref.store %r#0, %out[] : memref<i32>
  return
}

// The flag is yielded a carried value, not a constant.
func.func @flag_varies(%buf: memref<8xf64>, %out: memref<i32>, %go: i1) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %v = arith.constant 2.0 : f64
  %r:2 = scf.while (%flag = %true, %res = %c0_i32) : (i1, i32) -> (i32, i1) {
    %pair:2 = scf.if %flag -> (i32, i1) {
      affine.store %v, %buf[0] : memref<8xf64>
      %again = arith.andi %go, %true : i1
      scf.yield %c5_i32, %again : i32, i1
    } else {
      scf.yield %res, %false : i32, i1
    }
    scf.condition(%pair#1) %pair#0, %pair#1 : i32, i1
  } do {
  ^bb0(%res: i32, %again: i1):
    scf.yield %again, %res : i1, i32
  }
  memref.store %r#0, %out[] : memref<i32>
  return
}

// The selected branch yields a value that is not false.
func.func @branch_not_false(%buf: memref<8xf64>, %out: memref<i32>, %go: i1) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %v = arith.constant 2.0 : f64
  %r:2 = scf.while (%flag = %true, %res = %c0_i32) : (i1, i32) -> (i32, i1) {
    %pair:2 = scf.if %flag -> (i32, i1) {
      affine.store %v, %buf[0] : memref<8xf64>
      %again = arith.andi %go, %true : i1
      scf.yield %c5_i32, %again : i32, i1
    } else {
      scf.yield %res, %go : i32, i1
    }
    scf.condition(%pair#1) %pair#0, %pair#1 : i32, i1
  } do {
  ^bb0(%res: i32, %again: i1):
    scf.yield %false, %res : i1, i32
  }
  memref.store %r#0, %out[] : memref<i32>
  return
}

// CHECK:    func.func @flag_if(%arg0: memref<8xf64>, %arg1: memref<i32>, %arg2: i1) {
// CHECK-NEXT:    %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    affine.store %cst, %arg0[0] : memref<8xf64>
// CHECK-NEXT:    memref.store %c5_i32, %arg1[] : memref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @flag_true(%arg0: memref<8xf64>, %arg1: memref<i32>, %arg2: i1) {
// CHECK-NEXT:    %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    scf.while : () -> () {
// CHECK-NEXT:      affine.store %cst, %arg0[0] : memref<8xf64>
// CHECK-NEXT:      scf.condition(%arg2)
// CHECK-NEXT:    } do {
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.store %c5_i32, %arg1[] : memref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @flag_varies(%arg0: memref<8xf64>, %arg1: memref<i32>, %arg2: i1) {
// CHECK-NEXT:    %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    scf.while : () -> () {
// CHECK-NEXT:      affine.store %cst, %arg0[0] : memref<8xf64>
// CHECK-NEXT:      scf.condition(%arg2)
// CHECK-NEXT:    } do {
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.store %c5_i32, %arg1[] : memref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @branch_not_false(%arg0: memref<8xf64>, %arg1: memref<i32>, %arg2: i1) {
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = scf.while (%arg3 = %true, %arg4 = %c0_i32) : (i1, i32) -> i32 {
// CHECK-NEXT:      %1 = arith.select %arg3, %c5_i32, %arg4 : i32
// CHECK-NEXT:      scf.if %arg3 {
// CHECK-NEXT:        affine.store %cst, %arg0[0] : memref<8xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.condition(%arg2) %1 : i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg3: i32):
// CHECK-NEXT:      scf.yield %false, %arg3 : i1, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.store %0, %arg1[] : memref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
