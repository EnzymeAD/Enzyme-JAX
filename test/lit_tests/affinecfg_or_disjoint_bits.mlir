// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// An or with a constant that only sets bits the other operand is known to
// leave zero adds that constant: (e << 2) | 2 is e * 4 + 2, an affine
// index, and the read through it an affine read. (The loops carry a
// dependence, so that only the reads are under test.)
module {
  func.func @shifted(%p: memref<?xf64>, %nq: index, %ne: index) {
    %c2_i32 = arith.constant 2 : i32
    affine.for %e = 0 to %ne {
      affine.for %i = 0 to %nq {
        %0 = arith.index_cast %e : index to i32
        %1 = arith.shli %0, %c2_i32 : i32
        %2 = arith.ori %1, %c2_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %3, %nq : index
        %5 = arith.addi %4, %i : index
        %v = memref.load %p[%5] : memref<?xf64>
        %w = affine.load %p[%i + %e * symbol(%nq) + 1] : memref<?xf64>
        %s = arith.addf %v, %w : f64
        affine.store %s, %p[%i + %e * symbol(%nq)] : memref<?xf64>
      }
    }
    return
  }

  // the constant reaches a bit the shift leaves: not an add
  func.func @overlap(%p: memref<?xf64>, %nq: index, %ne: index) {
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    affine.for %e = 0 to %ne {
      affine.for %i = 0 to %nq {
        %0 = arith.index_cast %e : index to i32
        %1 = arith.shli %0, %c2_i32 : i32
        %2 = arith.ori %1, %c4_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %3, %nq : index
        %5 = arith.addi %4, %i : index
        %v = memref.load %p[%5] : memref<?xf64>
        %w = affine.load %p[%i + %e * symbol(%nq) + 1] : memref<?xf64>
        %s = arith.addf %v, %w : f64
        affine.store %s, %p[%i + %e * symbol(%nq)] : memref<?xf64>
      }
    }
    return
  }
}

// CHECK:  func.func @shifted(%arg0: memref<?xf64>, %arg1: index, %arg2: index) {
// CHECK-NEXT:    affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:      affine.for %arg4 = 0 to %arg1 {
// CHECK-NEXT:        %0 = affine.load %arg0[%arg4 + (%arg3 * 4 + 2) * symbol(%arg1)] : memref<?xf64>
// CHECK-NEXT:        %1 = affine.load %arg0[%arg4 + %arg3 * symbol(%arg1) + 1] : memref<?xf64>
// CHECK-NEXT:        %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:        affine.store %2, %arg0[%arg4 + %arg3 * symbol(%arg1)] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func.func @overlap(%arg0: memref<?xf64>, %arg1: index, %arg2: index) {
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:      affine.for %arg4 = 0 to %arg1 {
// CHECK-NEXT:        %0 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:        %1 = arith.shli %0, %c2_i32 : i32
// CHECK-NEXT:        %2 = arith.ori %1, %c4_i32 : i32
// CHECK-NEXT:        %3 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:        %4 = arith.muli %3, %arg1 : index
// CHECK-NEXT:        %5 = arith.addi %4, %arg4 : index
// CHECK-NEXT:        %6 = memref.load %arg0[%5] : memref<?xf64>
// CHECK-NEXT:        %7 = affine.load %arg0[%arg4 + %arg3 * symbol(%arg1) + 1] : memref<?xf64>
// CHECK-NEXT:        %8 = arith.addf %6, %7 : f64
// CHECK-NEXT:        affine.store %8, %arg0[%arg4 + %arg3 * symbol(%arg1)] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
