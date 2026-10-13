// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s

// An access through an if that merely multiplexes two views defined above
// it re-conditions at the access site: a fresh if with the same guard,
// each arm accessing its concrete view.

#set = affine_set<()[s0] : (s0 >= 1)>
func.func @affine_if_access(%a: memref<16xf64, 1>, %b: memref<16xf64, 1>, %s: index, %v: f64, %i: index) -> f64 {
  %m = affine.if #set()[%s] -> memref<16xf64, 1> {
    affine.yield %a : memref<16xf64, 1>
  } else {
    affine.yield %b : memref<16xf64, 1>
  }
  memref.store %v, %m[%i] : memref<16xf64, 1>
  %x = memref.load %m[%i] : memref<16xf64, 1>
  return %x : f64
}

// CHECK:  func.func @affine_if_access(%[[a:.+]]: memref<16xf64, 1>, %[[b:.+]]: memref<16xf64, 1>, %[[s:.+]]: index, %[[v:.+]]: f64, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  affine.if #set()[%[[s]]] {
// CHECK-NEXT:    memref.store %[[v]], %[[a]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    memref.store %[[v]], %[[b]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[x:.+]] = affine.if #set()[%[[s]]] -> f64 {
// CHECK-NEXT:    %[[l1:.+]] = memref.load %[[a]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:    affine.yield %[[l1]] : f64
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[l2:.+]] = memref.load %[[b]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:    affine.yield %[[l2]] : f64
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

func.func @scf_if_access(%a: memref<16xf64, 1>, %b: memref<16xf64, 1>, %c: i1, %v: f64, %i: index) {
  %m = scf.if %c -> memref<16xf64, 1> {
    scf.yield %a : memref<16xf64, 1>
  } else {
    scf.yield %b : memref<16xf64, 1>
  }
  memref.store %v, %m[%i] : memref<16xf64, 1>
  return
}

// CHECK:  func.func @scf_if_access(%[[a:.+]]: memref<16xf64, 1>, %[[b:.+]]: memref<16xf64, 1>, %[[c:.+]]: i1, %[[v:.+]]: f64, %[[i:.+]]: index) {
// CHECK-NEXT:  scf.if %[[c]] {
// CHECK-NEXT:    memref.store %[[v]], %[[a]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    memref.store %[[v]], %[[b]][%[[i]]] : memref<16xf64, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// -----

// A view produced inside an arm does not dominate the access site: the
// access stays where it is.

func.func @arm_defined_view(%p: !llvm.ptr, %q: !llvm.ptr, %c: i1, %v: f64, %i: index) {
  %m = scf.if %c -> memref<?xf64> {
    %mv = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    scf.yield %mv : memref<?xf64>
  } else {
    %mv = "enzymexla.pointer2memref"(%q) : (!llvm.ptr) -> memref<?xf64>
    scf.yield %mv : memref<?xf64>
  }
  memref.store %v, %m[%i] : memref<?xf64>
  return
}

// CHECK:  func.func @arm_defined_view(%[[p:.+]]: !llvm.ptr, %[[q:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[v:.+]]: f64, %[[i:.+]]: index) {
// CHECK-NEXT:  %[[m:.+]] = scf.if %[[c]] -> (memref<?xf64>) {
// CHECK:  memref.store %[[v]], %[[m]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:  return
// CHECK-NEXT:  }
