// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module {
  func.func private @opaque_cast(%arg0: memref<20x50x50xf64, 1>, %arg1: memref<20x50x50xf64, 1>, %arg2: memref<30x16xi64, 1>) {
    affine.parallel (%i, %x) = (0, 0) to (30, 16) {
      %lin = arith.index_castui %i : index to i64
      affine.store %lin, %arg2[%i, %x] : memref<30x16xi64, 1>
      %0 = affine.load %arg0[%i floordiv 3 + 5, 5, %x + (%i mod 3) * 16 + 5] : memref<20x50x50xf64, 1>
      affine.store %0, %arg1[%i floordiv 3 + 5, 5, %x + (%i mod 3) * 16 + 5] : memref<20x50x50xf64, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @opaque_cast
// CHECK: %[[C3:.+]] = arith.constant 3 : index
// CHECK: affine.parallel (%[[MAJ:.+]], %[[X:.+]], %[[MIN:.+]]) = (0, 0, 0) to (10, 16, 3) {
// CHECK: %[[LIN:.+]] = arith.muli %[[MAJ]], %[[C3]] overflow<nuw> : index
// CHECK-NEXT: %[[LIN2:.+]] = arith.addi %[[LIN]], %[[MIN]] : index
// CHECK-NEXT: %[[C:.+]] = arith.index_castui %[[LIN2]] : index to i64
// CHECK-NEXT: affine.store %[[C]]
// CHECK-NEXT: affine.load %{{.*}}[%[[MAJ]] + 5, 5, %[[X]] + %[[MIN]] * 16 + 5] : memref<20x50x50xf64, 1>
