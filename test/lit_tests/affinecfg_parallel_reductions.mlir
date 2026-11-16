// RUN: enzymexlamlir-opt %s --lower-affine --affine-cfg | FileCheck %s

// lower-affine writes an affine.parallel reduction as an scf.parallel that
// starts from the kind's identity and combines with one arith op in its
// scf.reduce; affine-cfg reads that back. A loop seeded with another value
// folds it in after the loop.

func.func @roundtrip(%arg0: memref<?xf64>) -> (f64, i32) {
  %0:2 = affine.parallel (%i) = (0) to (8) reduce ("addf", "muli") -> (f64, i32) {
    %v = affine.load %arg0[%i] : memref<?xf64>
    %c = arith.fptosi %v : f64 to i32
    affine.yield %v, %c : f64, i32
  }
  return %0#0, %0#1 : f64, i32
}

func.func @seeded(%arg0: memref<?xf64>, %init: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.parallel (%i) = (%c0) to (%c8) step (%c1) init (%init) -> f64 {
    %v = memref.load %arg0[%i] : memref<?xf64>
    scf.reduce(%v : f64) {
    ^bb0(%a: f64, %b: f64):
      %s = arith.mulf %a, %b : f64
      scf.reduce.return %s : f64
    }
  }
  return %0 : f64
}

// CHECK:    func.func @roundtrip(%arg0: memref<?xf64>) -> (f64, i32) {
// CHECK-NEXT:    %0:2 = affine.parallel (%arg1) = (0) to (8) reduce ("addf", "muli") -> (f64, i32) {
// CHECK-NEXT:      %1 = affine.load %arg0[%arg1] : memref<?xf64>
// CHECK-NEXT:      %2 = arith.fptosi %1 : f64 to i32
// CHECK-NEXT:      affine.yield %1, %2 : f64, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0, %0#1 : f64, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @seeded(%arg0: memref<?xf64>, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = affine.parallel (%arg2) = (0) to (8) reduce ("mulf") -> (f64) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:      affine.yield %2 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.mulf %arg1, %0 : f64
// CHECK-NEXT:    return %1 : f64
// CHECK-NEXT:  }
