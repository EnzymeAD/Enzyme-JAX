// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module {
  func.func @f(%arg0: memref<20x10xf64>) {
    %cst = arith.constant 10 : i64
    affine.parallel (%i) = (0) to (200) {
        %0 = affine.load %arg0[%i floordiv 10, %i mod 10] : memref<20x10xf64>
        %1 = arith.index_castui %i : index to i64
        %2 = arith.floordivsi %1, %cst : i64
        %3 = arith.sitofp %2 : i64 to f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg0[%i floordiv 10, %i mod 10] : memref<20x10xf64>
    }
    return
  }
}

// CHECK:  func.func @f(%[[MEMREF:.+]]: memref<20x10xf64>) {
// CHECK-NEXT:    affine.parallel (%[[IV1:.+]], %[[IV2:.+]]) = (0, 0) to (20, 10) {
// CHECK-NEXT:        %[[VAL:.+]] = affine.load %[[MEMREF]][%[[IV1]], %[[IV2]]] : memref<20x10xf64>
// CHECK-NEXT:        %[[IVINT:.+]] = arith.index_cast %[[IV1]] : index to i64
// CHECK-NEXT:        %[[IVFLOAT:.+]] = arith.sitofp %[[IVINT]] : i64 to f64
// CHECK-NEXT:        %[[VALFINAL:.+]] = arith.addf %[[VAL]], %[[IVFLOAT]] : f64
// CHECK-NEXT:        affine.store %[[VALFINAL]], %[[MEMREF]][%[[IV1]], %[[IV2]]] : memref<20x10xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
