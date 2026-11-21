// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s

// The store twin of LoadSelect: a store through a selected view splits into
// an if with one store per arm, each through a concrete view.

func.func @store_select(%a: memref<16xf64, 1>, %b: memref<16xf64, 1>, %c: i1, %v: f64, %i: index) {
  %m = arith.select %c, %a, %b : memref<16xf64, 1>
  memref.store %v, %m[%i] : memref<16xf64, 1>
  return
}

// CHECK:  func.func @store_select(%[[a:.+]]: memref<16xf64, 1>, %[[b:.+]]: memref<16xf64, 1>, %[[c:.+]]: i1, %[[v:.+]]: f64, %[[i:.+]]: index) {
// CHECK-NEXT:  scf.if %[[c]] {
// CHECK-NEXT:    memref.store %[[v]], %[[a]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    memref.store %[[v]], %[[b]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// -----

func.func @affine_store_select(%a: memref<16xf64, 1>, %b: memref<16xf64, 1>, %c: i1, %v: f64) {
  affine.parallel (%t) = (0) to (16) {
    %m = arith.select %c, %a, %b : memref<16xf64, 1>
    affine.store %v, %m[%t] : memref<16xf64, 1>
  }
  return
}

// CHECK:  func.func @affine_store_select(%[[a:.+]]: memref<16xf64, 1>, %[[b:.+]]: memref<16xf64, 1>, %[[c:.+]]: i1, %[[v:.+]]: f64) {
// CHECK-NEXT:  affine.parallel (%[[t:.+]]) = (0) to (16) {
// CHECK-NEXT:    scf.if %[[c]] {
// CHECK-NEXT:      affine.store %[[v]], %[[a]][%[[t]]] : memref<16xf64, 1>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.store %[[v]], %[[b]][%[[t]]] : memref<16xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }
