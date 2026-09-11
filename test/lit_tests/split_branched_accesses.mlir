// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(split-branched-accesses,canonicalize)" | FileCheck %s

// An access at an index a branch chose between constants is done in each arm
// instead, at the constant that arm chose. Arithmetic between the branch and
// the access is sunk into the arms beforehand by canonicalization
// (canonicalize_parallel_if_constants.mlir), so the index arrives here as the
// branch's own result.

#set = affine_set<()[s0] : (s0 >= 1)>
module {

  func.func @scf_load(%m: memref<?xi32>, %c: i1) -> i32 {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = scf.if %c -> index { scf.yield %c3 : index } else { scf.yield %c7 : index }
    %v = memref.load %m[%i] : memref<?xi32>
    return %v : i32
  }
  func.func @affine_store(%m: memref<?xf64>, %s: index, %val: f64) {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = affine.if #set()[%s] -> index { affine.yield %c3 : index } else { affine.yield %c7 : index }
    affine.store %val, %m[%i] : memref<?xf64>
    return
  }
  func.func @scf_store(%m: memref<?xi32>, %c: i1, %val: i32) {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = scf.if %c -> index { scf.yield %c3 : index } else { scf.yield %c7 : index }
    memref.store %val, %m[%i] : memref<?xi32>
    return
  }

  // an arm that does not choose a constant is left alone
  func.func @dynamic_arm(%m: memref<?xi32>, %c: i1, %d: index) -> i32 {
    %c3 = arith.constant 3 : index
    %i = scf.if %c -> index {
      scf.yield %c3 : index
    } else {
      scf.yield %d : index
    }
    %v = memref.load %m[%i] : memref<?xi32>
    return %v : i32
  }

  // the memref and the value stored are computed after the branch: the branch
  // is asked again where the access stands, so nothing has to move
  func.func @operands_after_branch(%p: !llvm.ptr, %s: index, %x: f64) {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = affine.if #set()[%s] -> index { affine.yield %c3 : index } else { affine.yield %c7 : index }
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    %v = arith.addf %x, %x : f64
    memref.store %v, %m[%i] : memref<?xf64>
    return
  }
}

// CHECK:  func.func @scf_load(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1) -> i32 {
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 7 : index
// CHECK-NEXT:  %[[v5:.+]] = scf.if %[[v2]] -> (i32) {
// CHECK-NEXT:  %[[v6:.+]] = memref.load %[[v1]][%[[v3]]] : memref<?xi32>
// CHECK-NEXT:  scf.yield %[[v6]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[v7:.+]] = memref.load %[[v1]][%[[v4]]] : memref<?xi32>
// CHECK-NEXT:  scf.yield %[[v7]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v5]] : i32
// CHECK-NEXT:  }

// CHECK:  func.func @affine_store(%[[v1:.+]]: memref<?xf64>, %[[v2:.+]]: index, %[[v3:.+]]: f64) {
// CHECK-NEXT:  affine.if #set()[%[[v2]]] {
// CHECK-NEXT:  affine.store %[[v3]], %[[v1]][3] : memref<?xf64>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.store %[[v3]], %[[v1]][7] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @scf_store(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1, %[[v3:.+]]: i32) {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v5:.+]] = arith.constant 7 : index
// CHECK-NEXT:  scf.if %[[v2]] {
// CHECK-NEXT:  memref.store %[[v3]], %[[v1]][%[[v4]]] : memref<?xi32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  memref.store %[[v3]], %[[v1]][%[[v5]]] : memref<?xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @dynamic_arm(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1, %[[v3:.+]]: index) -> i32 {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v5:.+]] = arith.select %[[v2]], %[[v4]], %[[v3]] : index
// CHECK-NEXT:  %[[v6:.+]] = memref.load %[[v1]][%[[v5]]] : memref<?xi32>
// CHECK-NEXT:  return %[[v6]] : i32
// CHECK-NEXT:  }

// CHECK:  func.func @operands_after_branch(%[[o1:.+]]: !llvm.ptr, %[[o2:.+]]: index, %[[o3:.+]]: f64) {
// CHECK-NEXT:  %[[o4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[o5:.+]] = arith.constant 7 : index
// CHECK-NEXT:  %[[o6:.+]] = "enzymexla.pointer2memref"(%[[o1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[o7:.+]] = arith.addf %[[o3]], %[[o3]] : f64
// CHECK-NEXT:  affine.if #set()[%[[o2]]] {
// CHECK-NEXT:  memref.store %[[o7]], %[[o6]][%[[o4]]] : memref<?xf64>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  memref.store %[[o7]], %[[o6]][%[[o5]]] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }
