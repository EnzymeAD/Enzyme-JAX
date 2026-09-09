// RUN: enzymexlamlir-opt --polygeist-mem2reg %s | FileCheck %s

// A read at a slot a branch picks between constants lands in one of the
// stored slots yet names none; it splits into the same branches around a
// read at each constant, and every leaf forwards. A three-way choice by a
// direction arrives as two nested affine branches, the shape of a kernel
// reading one of a closure's tensor dimensions.

#set0 = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (d0 - 1 == 0)>

func.func @nested(%d: index, %a: i32, %b: i32, %c: i32) -> i32 {
  %alloca = memref.alloca() : memref<8xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  memref.store %a, %alloca[%c1] : memref<8xi32>
  memref.store %b, %alloca[%c2] : memref<8xi32>
  memref.store %c, %alloca[%c5] : memref<8xi32>
  %inner = affine.if #set1(%d) -> index { affine.yield %c2 : index } else { affine.yield %c5 : index }
  %slot = affine.if #set0(%d) -> index { affine.yield %c1 : index } else { affine.yield %inner : index }
  %v = memref.load %alloca[%slot] : memref<8xi32>
  return %v : i32
}

func.func @select_over_branch(%d: index, %p: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %alloca = memref.alloca() : memref<8xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  memref.store %a, %alloca[%c1] : memref<8xi32>
  memref.store %b, %alloca[%c2] : memref<8xi32>
  memref.store %c, %alloca[%c5] : memref<8xi32>
  %inner = affine.if #set1(%d) -> index { affine.yield %c2 : index } else { affine.yield %c5 : index }
  %slot = arith.select %p, %c1, %inner : index
  %v = memref.load %alloca[%slot] : memref<8xi32>
  return %v : i32
}

// CHECK:  #set = affine_set<(d0) : (d0 - 1 == 0)>
// CHECK-NEXT:#set1 = affine_set<(d0) : (d0 == 0)>
// CHECK-NEXT:module {
// CHECK-NEXT:  func.func @nested(%[[a1:.+]]: index, %[[a2:.+]]: i32, %[[a3:.+]]: i32, %[[a4:.+]]: i32) -> i32 {
// CHECK-NEXT:    %[[a5:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[a6:.+]] = arith.constant 2 : index
// CHECK-NEXT:    %[[a7:.+]] = arith.constant 5 : index
// CHECK-NEXT:    %[[a8:.+]] = affine.if #set(%[[a1]]) -> index {
// CHECK-NEXT:      affine.yield %[[a6]] : index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a7]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a9:.+]] = affine.if #set1(%[[a1]]) -> index {
// CHECK-NEXT:      affine.yield %[[a5]] : index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a8]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a10:.+]] = affine.if #set1(%[[a1]]) -> i32 {
// CHECK-NEXT:      affine.yield %[[a2]] : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[a11:.+]] = affine.if #set(%[[a1]]) -> i32 {
// CHECK-NEXT:        affine.yield %[[a3]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %[[a4]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.yield %[[a11]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a10]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @select_over_branch(%[[a1]]: index, %[[a2]]: i1, %[[a3]]: i32, %[[a4]]: i32, %[[a12:.+]]: i32) -> i32 {
// CHECK-NEXT:    %[[a5]] = arith.constant 1 : index
// CHECK-NEXT:    %[[a6]] = arith.constant 2 : index
// CHECK-NEXT:    %[[a7]] = arith.constant 5 : index
// CHECK-NEXT:    %[[a8]] = affine.if #set(%[[a1]]) -> index {
// CHECK-NEXT:      affine.yield %[[a6]] : index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a7]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a9]] = arith.select %[[a2]], %[[a5]], %[[a8]] : index
// CHECK-NEXT:    %[[a10]] = scf.if %[[a2]] -> (i32) {
// CHECK-NEXT:      scf.yield %[[a3]] : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[a11]] = affine.if #set(%[[a1]]) -> i32 {
// CHECK-NEXT:        affine.yield %[[a4]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %[[a12]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[a11]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a10]] : i32
// CHECK-NEXT:  }
