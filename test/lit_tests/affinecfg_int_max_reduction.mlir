// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// A loop carrying the max of what it reads is a parallel reduction.
func.func @row_max(%off: memref<101xi32>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %i = 0 to 100 iter_args(%m = %c0_i32) -> (i32) {
    %lo = affine.load %off[%i] : memref<101xi32>
    %hi = affine.load %off[%i + 1] : memref<101xi32>
    %d = arith.subi %hi, %lo : i32
    %mx = arith.maxsi %m, %d : i32
    affine.yield %mx : i32
  }
  return %0 : i32
}

// CHECK:  func.func @row_max(%arg0: memref<101xi32>) -> i32 {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = affine.parallel (%arg1) = (0) to (100) reduce ("maxs") -> (i32) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg1] : memref<101xi32>
// CHECK-NEXT:      %3 = affine.load %arg0[%arg1 + 1] : memref<101xi32>
// CHECK-NEXT:      %4 = arith.subi %3, %2 : i32
// CHECK-NEXT:      affine.yield %4 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.maxsi %0, %c0_i32 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }

// -----

// The same loop lowered: an scf.parallel reducing with maxsi.
func.func @row_max_scf(%off: memref<101xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = scf.parallel (%i) = (%c0) to (%c100) step (%c1) init (%c0_i32) -> i32 {
    %lo = memref.load %off[%i] : memref<101xi32>
    %i1 = arith.addi %i, %c1 : index
    %hi = memref.load %off[%i1] : memref<101xi32>
    %d = arith.subi %hi, %lo : i32
    scf.reduce(%d : i32) {
    ^bb0(%a: i32, %b: i32):
      %m = arith.maxsi %a, %b : i32
      scf.reduce.return %m : i32
    }
  }
  return %0 : i32
}

// CHECK:  func.func @row_max_scf(%arg0: memref<101xi32>) -> i32 {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = affine.parallel (%arg1) = (0) to (100) reduce ("maxs") -> (i32) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg1] : memref<101xi32>
// CHECK-NEXT:      %3 = affine.load %arg0[%arg1 + 1] : memref<101xi32>
// CHECK-NEXT:      %4 = arith.subi %3, %2 : i32
// CHECK-NEXT:      affine.yield %4 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.maxsi %0, %c0_i32 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
