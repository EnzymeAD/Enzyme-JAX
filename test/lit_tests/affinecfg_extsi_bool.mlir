// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// A sign extension of an affine condition is an affine.if yielding -1 or 0,
// the way a zero extension yields 1 or 0.

func.func @extsi_dim(%d: i32, %out: memref<?xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.parallel (%t, %c) = (0, 0) to (8, 2) {
    %ci = arith.index_castui %c : index to i32
    %ne = arith.cmpi ne, %ci, %c0_i32 : i32
    %e = arith.extsi %ne : i1 to i32
    %v = arith.addi %d, %e : i32
    affine.store %v, %out[%t + 8 * %c] : memref<?xi32>
  }
  return
}

// CHECK:       #[[SET:.+]] = affine_set<(d0) : (d0 - 1 >= 0)>
// CHECK-LABEL: func.func @extsi_dim
// CHECK-DAG:     %[[M1:.+]] = arith.constant -1 : i32
// CHECK-DAG:     %[[Z:.+]] = arith.constant 0 : i32
// CHECK:         affine.parallel (%[[T:.+]], %[[C:.+]]) = (0, 0) to (8, 2)
// CHECK-NEXT:      %[[E:.+]] = affine.if #[[SET]](%[[C]]) -> i32 {
// CHECK-NEXT:        affine.yield %[[M1]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %[[Z]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[V:.+]] = arith.addi %arg0, %[[E]] : i32
// CHECK-NEXT:      affine.store %[[V]], %arg1[%[[T]] + %[[C]] * 8]
