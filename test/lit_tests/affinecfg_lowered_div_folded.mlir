// RUN: enzymexlamlir-opt --affine-cfg --canonicalize %s | FileCheck %s

// lower-affine's expansion of a bound `(1 - c) floordiv 2 + 3`, after a
// canonicalize: the dividend arm `-1 - x` of x = 1 - c is folded into
// `c - 2`. Read back whole as the floordiv, the loop it bounds raises.
func.func @floordiv(%arg0: memref<?xf64>) {
  %c-2 = arith.constant -2 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c-1 = arith.constant -1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  scf.for %arg1 = %c0 to %c3 step %c1 {
    %0 = arith.subi %c1, %arg1 : index
    %1 = arith.cmpi slt, %0, %c0 : index
    %2 = arith.addi %arg1, %c-2 : index
    %3 = arith.select %1, %2, %0 : index
    %4 = arith.divsi %3, %c2 : index
    %5 = arith.subi %c-1, %4 : index
    %6 = arith.select %1, %5, %4 : index
    %7 = arith.addi %6, %c3 : index
    scf.for %arg2 = %c0 to %7 step %c1 {
      %8 = arith.muli %arg1, %c8 overflow<nsw> : index
      %9 = arith.addi %arg2, %8 : index
      memref.store %cst, %arg0[%9] : memref<?xf64>
    }
  }
  return
}

// The same for `(1 - c) ceildiv 2 + 3`: `0 - x` is `-c` and `x - 1` is
// `c - 1` after the canonicalize.
func.func @ceildiv(%arg0: memref<?xf64>) {
  %c-1 = arith.constant -1 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  scf.for %arg1 = %c0 to %c3 step %c1 {
    %0 = arith.subi %c1, %arg1 : index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.addi %arg1, %c-1 : index
    %3 = arith.subi %c0, %arg1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c2 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    %9 = arith.addi %8, %c3 : index
    scf.for %arg2 = %c0 to %9 step %c1 {
      %10 = arith.muli %arg1, %c8 overflow<nsw> : index
      %11 = arith.addi %arg2, %10 : index
      memref.store %cst, %arg0[%11] : memref<?xf64>
    }
  }
  return
}

// CHECK: func.func @floordiv(%arg0: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.parallel (%arg1) = (0) to (3) {
// CHECK-NEXT:     affine.parallel (%arg2) = (0) to ((-%arg1 + 1) floordiv 2 + 3) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg2 + %arg1 * 8] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @ceildiv(%arg0: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.parallel (%arg1) = (0) to (3) {
// CHECK-NEXT:     affine.parallel (%arg2) = (0) to ((-%arg1 + 1) ceildiv 2 + 3) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg2 + %arg1 * 8] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
