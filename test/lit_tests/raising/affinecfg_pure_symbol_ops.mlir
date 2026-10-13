// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

module {
  func.func @ctlz_bound(%arg0: memref<?xf64>, %n: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant 1.0 : f64
    affine.parallel (%t) = (0) to (64) {
      %s = arith.shrui %n, %c1_i32 : i32
      %z = math.ctlz %s : i32
      %w = arith.subi %c32_i32, %z : i32
      %ub = arith.maxui %w, %c1_i32 : i32
      %ubi = arith.index_castui %ub : i32 to index
      scf.for %i = %c0 to %ubi step %c1 {
        %idx = arith.addi %i, %t : index
        memref.store %cst, %arg0[%idx] : memref<?xf64>
      }
    }
    return
  }
}

// The bound chain runs through math.ctlz, which is no arith op but is pure, so
// it is a valid symbol: the chain hoists out of the parallel and the loop
// becomes affine and merges into it.

// CHECK-LABEL:   func.func @ctlz_bound(
// CHECK-SAME:                          %[[A:.+]]: memref<?xf64>, %[[N:.+]]: i32) {
// CHECK-NEXT:      %[[CST:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:      %[[C32:.+]] = arith.constant 32 : i32
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[S:.+]] = arith.shrui %[[N]], %[[C1]] : i32
// CHECK-NEXT:      %[[Z:.+]] = math.ctlz %[[S]] : i32
// CHECK-NEXT:      %[[W:.+]] = arith.subi %[[C32]], %[[Z]] : i32
// CHECK-NEXT:      %[[UB:.+]] = arith.maxui %[[W]], %[[C1]] : i32
// CHECK-NEXT:      %[[UBI:.+]] = arith.index_cast %[[UB]] : i32 to index
// CHECK-NEXT:      affine.parallel (%[[T:.+]], %[[I:.+]]) = (0, 0) to (64, symbol(%[[UBI]])) {
// CHECK-NEXT:        affine.store %[[CST]], %[[A]][%[[I]] + %[[T]]] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
